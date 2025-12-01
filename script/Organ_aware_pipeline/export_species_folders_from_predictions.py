#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Organize accepted predictions into species/class folders.

- Edit USER_CONFIG below, then run this file (no CLI needed).
- If you pass CLI args, they override USER_CONFIG.

Dest layout: out_root/<Species> <Class>/<filename>
"""

import argparse
import os
import re
import sys
from types import SimpleNamespace
from typing import List
import shutil
import pandas as pd

# ========= USER CONFIG =========
USER_CONFIG = {
    # Path to your thresholded predictions CSV
    "pred_csv": r"E:/Santosh_master_thesis/prediction_metadata_LOT_thresholded.csv",
    # Output root where files will be copied/moved
    "out_root": r"E:/Santosh_master_thesis/classified_Leaves",
    # Only export these classes (exclude "Others" by not listing it)
    "include_classes": ["Leaves"],
    # "copy" or "move"
    "action": "copy",
    # Dry run (True = print actions only)
    "dry_run": False,
}
# =================================


def _species_from_path(path: str) -> str:
    # ...existing code...
    parent = os.path.basename(os.path.dirname(path))
    parent = re.sub(r"^\d+_", "", parent)  # strip leading numeric prefixes like 001_
    parent = parent.replace("_", " ")      # Abies_alba -> Abies alba
    return parent


def _parse_args():
    # Optional CLI (overrides USER_CONFIG if provided)
    ap = argparse.ArgumentParser(
        description="Export species/class folders from thresholded predictions CSV",
        add_help=True,
    )
    ap.add_argument("--pred-csv", help="Thresholded predictions CSV (with passed_threshold column)")
    ap.add_argument("--out-root", help="Output root directory")
    ap.add_argument("--include-classes", nargs="*", help="Classes to include (e.g., Leaves)")
    ap.add_argument("--action", choices=["copy", "move"], help="Copy or move files")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without copying/moving")
    args = ap.parse_args()
    return args


def _load_config() -> SimpleNamespace:
    # If no CLI args (only script path), use USER_CONFIG
    if len(sys.argv) == 1:
        cfg = {**USER_CONFIG}
    else:
        args = _parse_args()
        cfg = {**USER_CONFIG}
        if args.pred_csv is not None:
            cfg["pred_csv"] = args.pred_csv
        if args.out_root is not None:
            cfg["out_root"] = args.out_root
        if args.include_classes is not None:
            cfg["include_classes"] = args.include_classes
        if args.action is not None:
            cfg["action"] = args.action
        if args.dry_run:
            cfg["dry_run"] = True
    # Normalize types
    cfg["include_classes"] = [str(c).strip() for c in cfg["include_classes"]]
    cfg["action"] = cfg["action"].lower()
    return SimpleNamespace(**cfg)


def run_export(cfg: SimpleNamespace) -> int:
    if not os.path.isfile(cfg.pred_csv):
        raise FileNotFoundError(f"pred_csv not found: {cfg.pred_csv}")
    os.makedirs(cfg.out_root, exist_ok=True)

    df = pd.read_csv(cfg.pred_csv)

    # Accept either boolean or string-y booleans; if column missing, assume all True
    if "passed_threshold" in df.columns:
        passed = df["passed_threshold"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        passed = pd.Series([True] * len(df), index=df.index)

    required_cols = {"image_path", "predicted_class"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV missing required columns {required_cols}; found: {list(df.columns)}")

    keep = df[passed & (df["predicted_class"].astype(str).isin(cfg.include_classes))].copy()
    if keep.empty:
        print("No rows matched include classes and passed_threshold=True.")
        return 0

    n_total, n_missing = 0, 0
    for _, row in keep.iterrows():
        src = str(row["image_path"]).strip()
        cls = str(row["predicted_class"]).strip()
        species = _species_from_path(src)
        fname = os.path.basename(src)

        dst_dir = os.path.join(cfg.out_root, f"{species} {cls}")  # e.g., "Abies alba Leaves"
        dst = os.path.join(dst_dir, fname)

        if not os.path.isfile(src):
            n_missing += 1
            continue

        if not cfg.dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            if cfg.action == "copy":
                shutil.copy2(src, dst)
            else:
                shutil.move(src, dst)
        n_total += 1

    print(f"Done. {cfg.action}ed {n_total} files into: {cfg.out_root}")
    if n_missing:
        print(f"Skipped {n_missing} missing source files.")
    return 0


def main():
    cfg = _load_config()
    return run_export(cfg)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)