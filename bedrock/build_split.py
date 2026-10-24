"""Bedrock: build the group table, the final holdout and the 5-fold cross-validation pool.

Inputs (metadata only, no expression or images):
  --split  split_v3.2 CSV (one row per HEST sample: study_key, dup_group, stbank_overlap, eligibility, old split)
  --meta   HEST_v1_3_0.csv (HEST metadata: dataset_title, patient, subseries, organ, species, st_technology)

Rules (written before any output was looked at):
  1. Eligible = Adnan's 1038 slides (broad-transcriptome panel, human or mouse).
  2. Groups = union of: same study_key; same dup_group (duplicate sections / shared H&E, from the 2026-10-19 check);
     same (study_key, patient) when patient is filled; and for 10x Genomics demo data (id TENX*), same
     (species, organ), because 10x demos of one tissue come from one company and are often adjacent sections.
  3. A group that contains any slide seen in OmiCLIP pretraining (stbank_overlap) is train-only: it never enters
     an evaluation fold or the holdout.
  4. Final holdout ~20% of the evaluable groups, stratified by species x platform (organ x species x platform
     leaves most strata with < 3 groups): in every stratum with >= 3 groups, ceil(0.2 * n) groups are drawn.
     Only groups holding <= 5% of the eligible spots can be drawn, so one huge study cannot leave training.
     Groups of the old locked holdout (split_v3) that were never opened and contain no slide from the old
     val / test are kept in the holdout first. (Revised 2026-10-24 after the first build gave only 6 holdout
     groups; no model output exists for this split.)
  5. The remaining evaluable groups get 5 folds (StratifiedGroupKFold on the same strata, shuffled, fixed seed).
     Train-only groups get fold = -1 (they are in the train part of every fold).
  6. Each fold's train part gets a checkpoint-selection subset: ~10% of its evaluable groups.
"""
import argparse
import hashlib
import math

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 20261024
N_FOLDS = 5
HOLDOUT_FRAC = 0.20
HOLDOUT_MAX_SPOT_FRAC = 0.05
SELECT_FRAC = 0.10


class UnionFind:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def link_by(uf, df, cols):
    for _, ids in df.dropna(subset=cols).groupby(cols)["id"]:
        ids = list(ids)
        for other in ids[1:]:
            uf.union(ids[0], other)


def build_groups(d):
    uf = UnionFind(d["id"])
    link_by(uf, d, ["study_key"])
    link_by(uf, d, ["dup_group"])
    pat = d.assign(patient=d["patient"].str.strip().str.lower().replace("", np.nan))
    link_by(uf, pat, ["study_key", "patient"])
    tenx = d[d["id"].str.startswith("TENX")]
    link_by(uf, tenx, ["species", "organ"])
    root = d["id"].map(uf.find)
    names = {r: f"G{i:03d}" for i, r in enumerate(sorted(root.unique()))}
    return root.map(names)


def stratum(d):
    return d["organ"].astype(str) + "|" + d["species"].astype(str) + "|" + d["platform"].astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s = pd.read_csv(a.split)
    m = pd.read_csv(a.meta)[["id", "patient", "subseries", "dataset_title"]]
    d = s[s["eligible_adnan"]].merge(m, on="id", how="left")
    d["group"] = build_groups(d)
    d["stratum"] = stratum(d)
    d["coarse"] = d["species"].astype(str) + "|" + d["platform"].astype(str)

    g = d.groupby("group").agg(
        stbank=("stbank_overlap", "any"),
        stratum=("stratum", lambda x: x.mode().iloc[0]),  # majority stratum of the group
        n_slides=("id", "size"),
        coarse=("coarse", lambda x: x.mode().iloc[0]),
        spots=("n_spots", "sum"),
        old_holdout_clean=("split_v3_2", lambda x: (x == "holdout").all()),
        touched=("split_v3_2", lambda x: x.isin(["val", "test"]).any()),
    )
    g["train_only"] = g["stbank"]
    evaluable = g[~g["train_only"]].copy()

    rng = np.random.default_rng(SEED)
    holdout = set(evaluable.index[evaluable["old_holdout_clean"] & ~evaluable["touched"]])
    small = evaluable["spots"] <= HOLDOUT_MAX_SPOT_FRAC * d["n_spots"].sum()
    for st, grp in evaluable.groupby("coarse"):
        n = len(grp)
        if n < 3:
            continue
        need = math.ceil(HOLDOUT_FRAC * n) - len(holdout & set(grp.index))
        pool = sorted(set(grp.index[small[grp.index]]) - holdout)
        if need > 0:
            holdout |= set(rng.choice(pool, size=min(need, len(pool)), replace=False))
    g["role"] = np.where(g["train_only"], "train_only", np.where(g.index.isin(holdout), "holdout", "cv"))

    cv = g[g["role"] == "cv"]
    # strata with fewer groups than folds cannot be stratified; pool them
    counts = cv["stratum"].value_counts()
    strat = cv["stratum"].where(cv["stratum"].map(counts) >= N_FOLDS, "rare")
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    g["fold"] = -1
    for k, (_, te) in enumerate(sgkf.split(cv, strat, groups=cv.index)):
        g.loc[cv.index[te], "fold"] = k

    # checkpoint-selection subset per fold, drawn from that fold's train-side evaluable groups
    for k in range(N_FOLDS):
        train_side = sorted(g.index[(g["role"] == "cv") & (g["fold"] != k)])
        pick = rng.choice(train_side, size=max(1, round(SELECT_FRAC * len(train_side))), replace=False)
        g[f"select_f{k}"] = g.index.isin(pick)

    out = d.merge(g.drop(columns=["stratum", "coarse", "spots"]), left_on="group", right_index=True)
    cols = ["id", "group", "role", "fold"] + [f"select_f{k}" for k in range(N_FOLDS)] + [
        "study_key", "organ", "species", "platform", "stratum", "n_spots", "stbank_overlap", "split_v3_2", "dup_group"]
    out = out[cols].sort_values(["group", "id"])
    out.to_csv(a.out, index=False)
    md5 = hashlib.md5(open(a.out, "rb").read()).hexdigest()
    print(f"wrote {a.out} ({len(out)} slides, {out.group.nunique()} groups) md5 {md5}")


if __name__ == "__main__":
    main()
