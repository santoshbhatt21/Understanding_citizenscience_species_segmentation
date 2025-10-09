#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate ground-truth CSV from a labeled folder tree (class subfolders).

Output CSV schema:
    image_path,true_class

Examples (PowerShell):
    python "make_gt_csv_from_folder.py" \
      --labeled-root "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Label_Leaves_Others_Trunks_manual_500_images" \
      --out "E:/Santosh_master_thesis/val_ground_truth.csv"

    # Optional: write paths relative to a base (makes the CSV portable)
    python "make_gt_csv_from_folder.py" \
      --labeled-root "E:/.../Label_Leaves_Others_Trunks_manual_500_images" \
      --out "E:/.../val_ground_truth.csv" \
      --relative-to "E:/.../"

    # Optional: only include files that appear in a predictions CSV
    python "make_gt_csv_from_folder.py" \
      --labeled-root "E:/.../Label_Leaves_Others_Trunks_manual_500_images" \
      --out "E:/.../val_ground_truth_only_pred.csv" \
      --intersect-pred-csv "E:/.../prediction_metadata_Leaves_Others_Trunks.csv"
"""

import os
import argparse
import csv
from typing import Iterable, List, Set

import pandas as pd


def _parse_args():
    ap = argparse.ArgumentParser(description="Create ground-truth CSV from a labeled folder.")
    ap.add_argument("--labeled-root", required=True, help="Root folder with class subfolders (ImageFolder-style)")
    ap.add_argument("--out", required=True, help="Output CSV path (image_path,true_class)")
    ap.add_argument("--relative-to", default=None, help="If set, write image_path relative to this base path")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png,.bmp,.webp", help="Comma-separated list of image extensions")
    ap.add_argument("--intersect-pred-csv", default=None, help="Optional predictions CSV to intersect with (keeps only rows present there)")
    ap.add_argument("--pred-image-col", default="image_path", help="Column name in predictions CSV for the image path")
    return ap.parse_args()


def _walk_images(root: str, exts: Iterable[str]) -> List[str]:
    out: List[str] = []
    lex = tuple(e.lower() for e in exts)
    for d, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(lex):
                out.append(os.path.join(d, fn))
    return out


def _top_level_class(root: str, path: str) -> str:
    rel = os.path.relpath(path, root)
    parts = rel.split(os.sep)
    if len(parts) < 2:
        # Image directly under root; treat parent folder name as class if possible
        return os.path.basename(root)
    return parts[0]


def _normalize_paths_for_intersect(paths: Iterable[str]) -> Set[str]:
    s: Set[str] = set()
    for p in paths:
        q = os.path.normcase(os.path.normpath(str(p)))
        s.add(q)
    return s


def main():
    args = _parse_args()
    exts = [e.strip() for e in args.exts.split(",") if e.strip()]
    labeled_root = os.path.normpath(args.labeled_root)
    base_for_rel = os.path.normpath(args.relative_to) if args.relative_to else None

    # Optional intersection set from predictions CSV
    keep_abs: Set[str] = set()
    keep_base: Set[str] = set()
    if args.intersect_pred_csv:
        try:
            pred_df = pd.read_csv(args.intersect_pred_csv)
            if args.pred_image_col not in pred_df.columns:
                raise KeyError(f"Column '{args.pred_image_col}' not in predictions CSV")
            pred_paths = pred_df[args.pred_image_col].astype(str).tolist()
            keep_abs = _normalize_paths_for_intersect(pred_paths)
            keep_base = set(os.path.basename(p) for p in pred_paths)
        except Exception as e:
            raise RuntimeError(f"Failed to read predictions CSV for intersection: {e}")

    # Gather images and emit rows
    rows: List[dict] = []
    images = _walk_images(labeled_root, exts)
    for p in images:
        cls = _top_level_class(labeled_root, p)
        out_path = p
        if base_for_rel:
            try:
                out_path = os.path.relpath(p, base_for_rel)
            except Exception:
                out_path = p

        if keep_abs or keep_base:
            norm_p = os.path.normcase(os.path.normpath(p))
            base = os.path.basename(p)
            if (norm_p not in keep_abs) and (base not in keep_base):
                continue

        rows.append({"image_path": out_path, "true_class": cls})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "true_class"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
