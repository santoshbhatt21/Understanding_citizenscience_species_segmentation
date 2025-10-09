#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sample a reduced, balanced subset from a predictions CSV and emit a GT template CSV.

Input (predictions CSV): must contain columns: image_path, predicted_class, confidence

Outputs:
  - out_csv: CSV with columns image_path,true_class (true_class left blank for manual fill)
  - out_list (optional): text file listing sampled image paths

Examples (PowerShell):
  python "sample_predictions_for_gt.py" `
    --pred-csv "E:/Santosh_master_thesis/prediction_metadata_Leaves_Others_Trunks.csv" `
    --out-csv  "E:/Santosh_master_thesis/gt_template_3x100.csv" `
    --per-class 100 `
    --seed 42

After filling in true_class, use that CSV as --gt-csv in sweep_thresholds_from_predictions.py
"""

import argparse
import os
import sys
from typing import List, Optional

import pandas as pd


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Sample a subset from predictions and emit a ground-truth template CSV.")
    ap.add_argument("--pred-csv", required=True,
                    help="Predictions CSV with image_path,predicted_class,confidence")
    ap.add_argument("--out-csv", required=True,
                    help="Output template CSV (image_path,true_class)")
    ap.add_argument("--out-list", default=None,
                    help="Optional text file to list sampled image paths")
    ap.add_argument("--per-class", type=int, default=100,
                    help="Number of samples per predicted class")
    ap.add_argument("--classes", nargs="*", default=None,
                    help="Optional subset of classes to include (names)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling")
    return ap.parse_args()


def main():
    args = _parse_args()
    df = pd.read_csv(args.pred_csv)

    required = {"image_path", "predicted_class", "confidence"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns in predictions CSV. Need: {required}")

    if args.classes:
        df = df[df["predicted_class"].astype(str).isin(args.classes)].copy()
        if df.empty:
            raise RuntimeError("No rows remain after filtering by --classes")

    # Stratified sampling per predicted class
    sampled_paths: List[str] = []
    for cls, grp in df.groupby("predicted_class", dropna=False):
        n = min(args.per_class, len(grp))
        if n <= 0:
            continue
        sub = grp.sample(n=n, random_state=args.seed)
        sampled_paths.extend(sub["image_path"].tolist())

    # Build GT template
    out_df = pd.DataFrame({"image_path": sampled_paths,
                          "true_class": [""] * len(sampled_paths)})
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    if args.out_list:
        with open(args.out_list, "w", encoding="utf-8") as f:
            for p in sampled_paths:
                f.write(str(p) + "\n")

    print(f"Wrote template with {len(out_df)} rows to {args.out_csv}")
    if args.out_list:
        print(f"Wrote list to {args.out_list}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
