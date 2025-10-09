#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a ground-truth CSV from an ImageFolder-style labeled directory.

Output CSV columns:
    image_path,true_class

Usage (PowerShell example):
    py .\make_gt_csv_from_imagefolder.py `
      --labeled-data-path "E:\Santosh_master_thesis\Understanding_citizenscience_species_segmentation\Label_Leaves_Others_Trunks_manual_500_images" `
      --output-csv-path "E:\Santosh_master_thesis\val_ground_truth.csv"

Notes
- By default, paths are ABSOLUTE to maximize merge success with your predictions CSV.
- You can store RELATIVE paths instead with --use-rel-paths; if you do, be sure to align the same style when merging.
- Recognized extensions: .jpg, .jpeg, .png, .bmp, .webp .
"""

import os
import csv
import argparse
from typing import List, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def _parse_args():
    ap = argparse.ArgumentParser(description="Create ground-truth CSV from labeled ImageFolder directory.")
    ap.add_argument("--labeled-data-path", required=True, help="Root folder with class subfolders (ImageFolder style).")
    ap.add_argument("--output-csv-path", required=True, help="Where to save the CSV (image_path,true_class).")
    ap.add_argument("--max-per-class", type=int, default=None, help="Optional cap per class (random order of OS walk).")
    ap.add_argument("--use-rel-paths", action="store_true", help="Store paths relative to the labeled root instead of absolute.")
    return ap.parse_args()

def _iter_images_for_class(class_dir: str) -> List[str]:
    paths = []
    for d, _, files in os.walk(class_dir):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(d, fn))
    return paths

def main():
    args = _parse_args()
    root = os.path.abspath(args.labeled_data_path)
    rows: List[Tuple[str,str]] = []

    # classes = top-level dirs sorted for determinism (same as ImageFolder does)
    classes = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    if not classes:
        raise SystemExit(f"No class subfolders found under: {root}")

    for cls in classes:
        cdir = os.path.join(root, cls)
        imgs = _iter_images_for_class(cdir)
        if args.max_per_class is not None:
            imgs = imgs[:args.max_per_class]
        for p in imgs:
            path_out = os.path.relpath(p, root) if args.use_rel_paths else os.path.abspath(p)
            rows.append((path_out, cls))

    # Save
    os.makedirs(os.path.dirname(args.output_csv_path) or ".", exist_ok=True)
    with open(args.output_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "true_class"])
        for r in rows:
            w.writerow(r)

    print(f"✅ Wrote {len(rows)} rows to: {args.output_csv_path}")
    print(f"Classes discovered ({len(classes)}): {classes}")

if __name__ == "__main__":
    main()
