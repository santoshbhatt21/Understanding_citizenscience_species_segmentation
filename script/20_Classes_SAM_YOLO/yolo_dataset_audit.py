#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Segmentation Dataset Auditor
- Verifies 1:1 image↔label pairing for train/val/test.
- Finds images without labels and labels without images.
- Validates YOLO-seg polygon lines (class + x1 y1 ... xN yN).
- Optional class index check via --nc or data.yaml.
- Reports out-of-range coords and degenerate polygons.
- Writes a CSV with all issues and prints a compact summary.

Usage:
  python yolo_dataset_audit.py --root /path/to/dataset --splits train val --nc 20
  # or let it auto-detect nc from data.yaml (if present):
  python yolo_dataset_audit.py --root /path/to/dataset --splits train val

Expected folder layout:
  dataset/
    images/train/...
    labels/train/...
    images/val/...
    labels/val/...
    data.yaml   (optional; for nc/names)

"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import glob
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def read_yaml_nc(yaml_path: str) -> Optional[int]:
    """Tiny YAML reader to extract nc: <int> without external deps."""
    if not os.path.isfile(yaml_path):
        return None
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # tolerate "nc: 20" or "nc : 20"
                if line.lower().startswith("nc"):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val = parts[1].strip().split()[0]
                        return int(val)
    except Exception:
        return None
    return None

def list_images(root: str) -> List[str]:
    return [p for p in glob.glob(os.path.join(root, "**", "*.*"), recursive=True)
            if os.path.splitext(p)[1].lower() in IMG_EXTS]

def relpath_noext(path: str, base: str) -> str:
    r = os.path.relpath(path, base)
    return os.path.splitext(r)[0].replace("\\", "/")

def pairs(seq: List[float]) -> List[Tuple[float, float]]:
    return list(zip(seq[0::2], seq[1::2]))

def poly_area_norm(pts: List[Tuple[float, float]]) -> float:
    """Shoelace area in normalized coordinates."""
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i+1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5

def audit_split(root: str, split: str, nc: Optional[int], tiny_area: float):
    img_dir = os.path.join(root, "images", split)
    lbl_dir = os.path.join(root, "labels", split)
    images = list_images(img_dir)

    issues = []  # tuples: (issue, split, path, detail)
    seen_label_files = set()

    # Check images -> labels
    for img in images:
        base_rel = relpath_noext(img, img_dir)
        lbl = os.path.join(lbl_dir, base_rel + ".txt")
        if not os.path.isfile(lbl):
            issues.append(("missing_label", split, img, ""))
            continue
        seen_label_files.add(os.path.normpath(lbl))

        # Validate label content
        try:
            with open(lbl, "r", encoding="utf-8") as f:
                lines = f.readlines()
            if not lines:
                issues.append(("empty_label_file", split, lbl, ""))
                continue
            for ln, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    issues.append(("empty_line", split, f"{lbl}:{ln}", ""))
                    continue
                parts = line.split()
                # class id
                try:
                    cls = int(parts[0])
                except Exception as e:
                    issues.append(("bad_class_format", split, f"{lbl}:{ln}", line))
                    continue
                if nc is not None and (cls < 0 or cls >= nc):
                    issues.append(("class_oob", split, f"{lbl}:{ln}", f"class={cls}, nc={nc}"))

                # coords
                try:
                    nums = list(map(float, parts[1:]))
                except Exception:
                    issues.append(("non_numeric_coords", split, f"{lbl}:{ln}", line))
                    continue

                if len(nums) < 6 or len(nums) % 2 != 0:
                    issues.append(("bad_poly_len", split, f"{lbl}:{ln}", f"coords={len(nums)}"))
                    continue

                pts = pairs(nums)

                # range check
                bad_xy = [(x, y) for (x, y) in pts if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)]
                if bad_xy:
                    issues.append(("out_of_range_xy", split, f"{lbl}:{ln}", f"e.g. {bad_xy[:3]}"))

                # area check
                area = poly_area_norm(pts)
                if area < tiny_area:
                    issues.append(("tiny_or_degenerate", split, f"{lbl}:{ln}", f"area={area:.2e}"))
        except Exception as e:
            issues.append(("read_error", split, lbl, str(e)))

    # Check labels -> images (orphans)
    label_files = [p for p in glob.glob(os.path.join(lbl_dir, "**", "*.txt"), recursive=True)]
    for lbl in label_files:
        if os.path.normpath(lbl) not in seen_label_files:
            # find matching image
            base_rel = relpath_noext(lbl, lbl_dir)
            found = False
            for ext in IMG_EXTS:
                cand = os.path.join(img_dir, base_rel + ext)
                if os.path.isfile(cand):
                    found = True
                    break
            if not found:
                issues.append(("orphan_label", split, lbl, ""))

    return issues, len(images), len(label_files)

def write_csv(issues, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["issue", "split", "path", "detail"])
        for row in issues:
            w.writerow(row)

def main():
    ap = argparse.ArgumentParser(description="YOLO Segmentation Dataset Auditor")
    ap.add_argument("--root", required=True, help="Dataset root containing images/ and labels/")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to audit (default: train val)")
    ap.add_argument("--nc", type=int, default=None, help="Number of classes (overrides data.yaml)")
    ap.add_argument("--tiny-area", type=float, default=1e-5, help="Min polygon area (normalized) before flagging")
    ap.add_argument("--out", default="audit_report.csv", help="CSV path for issues")
    args = ap.parse_args()

    root = args.root
    if args.nc is None:
        yaml_nc = read_yaml_nc(os.path.join(root, "data.yaml"))
        nc = yaml_nc
    else:
        nc = args.nc

    all_issues = []
    summary = []
    for sp in args.splits:
        issues, nimg, nlbl = audit_split(root, sp, nc, args.tiny_area)
        all_issues.extend(issues)
        summary.append((sp, nimg, nlbl))

    write_csv(all_issues, args.out)

    # Print summary
    print("=" * 60)
    print("YOLO Dataset Audit Summary")
    print(f"Root: {root}")
    if nc is not None:
        print(f"Classes (nc): {nc}")
    else:
        print("Classes (nc): not provided / not found in data.yaml")
    print("-" * 60)
    for sp, nimg, nlbl in summary:
        print(f"[{sp}] images: {nimg:>6} | labels: {nlbl:>6} | Δ (img - lbl): {nimg - nlbl:+d}")
    print("-" * 60)

    # Issue breakdown
    cnt = Counter([t[0] for t in all_issues])
    if cnt:
        print("Issue counts:")
        for k, v in cnt.most_common():
            print(f"  {k:<22} : {v}")
    else:
        print("No issues found. ✅")

    print("-" * 60)
    print(f"Issues CSV: {os.path.abspath(args.out)}")
    print("Tip: Remove or fix 'missing_label' images before training,")
    print("     and ensure class indices match your nc/names mapping.")

if __name__ == "__main__":
    sys.exit(main())
