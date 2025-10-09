#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apply thresholds from a sweep JSON to a predictions CSV.

Inputs:
  - predictions CSV with columns: image_path, predicted_class, confidence
  - sweep JSON with keys: GLOBAL_MIN (float), PER_CLASS_MIN (dict)

Outputs:
  - writes an updated CSV with columns threshold_used and passed_threshold recomputed

Usage (PowerShell):
  python "apply_thresholds_from_json.py" `
    --pred-csv "E:/Santosh_master_thesis/prediction_metadata_Leaves_Others_Trunks.csv" `
    --sweep-json "E:/Santosh_master_thesis/threshold_sweep_LOT_subset_recommended.json" `
    --out-csv "E:/Santosh_master_thesis/prediction_metadata_Leaves_Others_Trunks_thresholded.csv"
"""

import argparse
import json
import os
import sys
import pandas as pd


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Apply thresholds from sweep JSON to predictions CSV")
    ap.add_argument("--pred-csv", required=True, help="Input predictions CSV")
    ap.add_argument("--sweep-json", required=True,
                    help="Recommended thresholds JSON from sweep")
    ap.add_argument("--out-csv", required=True, help="Output CSV path")
    ap.add_argument("--pred-class-col", default="predicted_class",
                    help="Predictions class column name")
    ap.add_argument("--pred-conf-col", default="confidence",
                    help="Predictions confidence column name")
    return ap.parse_args()


def main():
    args = _parse_args()
    df = pd.read_csv(args.pred_csv)
    with open(args.sweep_json, "r", encoding="utf-8") as f:
        rec = json.load(f)

    global_min = float(rec.get("GLOBAL_MIN", 0.9))
    per_class = rec.get("PER_CLASS_MIN", {}) or {}

    # Compute per-row threshold
    def _thr(row):
        cls = str(row.get(args.pred_class_col, ""))
        try:
            return float(per_class.get(cls, global_min))
        except Exception:
            return float(global_min)

    df["threshold_used"] = df.apply(_thr, axis=1)
    # Apply threshold; ensure numeric confidence
    df[args.pred_conf_col] = pd.to_numeric(
        df[args.pred_conf_col], errors="coerce")
    df["passed_threshold"] = (df[args.pred_conf_col]
                              >= df["threshold_used"]).astype(bool)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote updated predictions with thresholds to: {args.out_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
