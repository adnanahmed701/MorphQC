"""Checks every Bedrock split must pass. Run after any new split; exits non-zero on failure.

  1. No slide seen in OmiCLIP pretraining sits in the holdout or in an evaluation fold.
  2. No group is spread over more than one role or fold (so duplicate / adjacent sections never cross sets).
  3. Each fold's checkpoint-selection subset never overlaps that fold's evaluation part.
"""
import sys

import pandas as pd


def check(df):
    errors = []
    bad = df[df["stbank_overlap"] & (df["role"] != "train_only")]
    if len(bad):
        errors.append(f"{len(bad)} OmiCLIP-seen slides outside train_only: {list(bad.id[:5])}")
    spread = df.groupby("group")[["role", "fold"]].nunique()
    spread = spread[(spread["role"] > 1) | (spread["fold"] > 1)]
    if len(spread):
        errors.append(f"{len(spread)} groups split across roles/folds: {list(spread.index[:5])}")
    for c in [c for c in df.columns if c.startswith("select_f")]:
        k = int(c[len("select_f"):])
        clash = df[df[c] & (df["fold"] == k)]
        if len(clash):
            errors.append(f"fold {k}: {len(clash)} selection slides inside the evaluation fold")
    return errors


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    errs = check(df)
    for e in errs:
        print("FAIL", e)
    print("PASS" if not errs else f"{len(errs)} check(s) failed")
    sys.exit(1 if errs else 0)
