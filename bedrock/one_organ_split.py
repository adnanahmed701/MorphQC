"""One-organ branch (mouse brain Visium) on top of the Bedrock split.

Uses the same groups as the main split, so a slide is never in the evaluation part of one split and the
training part of the other for the same model family. Train-only and holdout roles are inherited. The
evaluable groups (cv groups with at least one scorable slide in this subset) get leave-one-group-out folds,
because there are too few groups for 5 folds. Unscorable slides (OmiCLIP relatives) are kept for training.
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="bedrock_split CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--species", default="Mus musculus")
    ap.add_argument("--organ", default="Brain")
    ap.add_argument("--platform", default="Visium")
    a = ap.parse_args()

    d = pd.read_csv(a.split)
    o = d[(d.species == a.species) & (d.organ == a.organ) & (d.platform == a.platform)].copy()
    cv_groups = sorted(o.loc[(o.role == "cv") & o.scorable, "group"].unique())
    o["fold_one_organ"] = o["group"].map({g: k for k, g in enumerate(cv_groups)}).fillna(-1).astype(int)
    o[["id", "group", "role", "fold_one_organ", "scorable", "study_key", "n_spots", "stbank_overlap"]].to_csv(a.out, index=False)
    print(f"{len(o)} slides, {o.group.nunique()} groups, {len(cv_groups)} evaluable groups -> {a.out}")
    print(o.groupby(["role", "group"]).agg(slides=("id", "size"), scorable=("scorable", "sum")).to_string())


if __name__ == "__main__":
    main()
