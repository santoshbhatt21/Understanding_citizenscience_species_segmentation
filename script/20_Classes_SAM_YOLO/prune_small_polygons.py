#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prune tiny/malformed YOLO segmentation polygons from labels to improve training quality.

For each labels/<split>/*.txt:
 - Drops polygons whose normalized area is below --min-area (default 1e-4)
 - Optionally clamps coordinates to [0,1]
 - Optionally limits max polygons per image (--max-per-image)
 - Skips malformed lines (odd coords, < 3 points)

Usage (PowerShell):
  python script/20_Classes_SAM_YOLO/prune_small_polygons.py \
    --root "E:/Santosh_master_thesis/DATA_YOLO11_strict_CAM_setting" --splits train val --min-area 1e-4 --clamp --backup
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Optional


def list_label_files(root: str, split: str) -> List[str]:
    target = os.path.join(root, "labels", split)
    out: List[str] = []
    for dirpath, _, filenames in os.walk(target):
        for fn in filenames:
            if fn.lower().endswith(".txt"):
                out.append(os.path.join(dirpath, fn))
    return out


def parse_line(line: str) -> Optional[Tuple[int, List[float]]]:
    parts = line.strip().split()
    if not parts:
        return None
    try:
        cls = int(float(parts[0]))
    except Exception:
        return None
    try:
        nums = list(map(float, parts[1:]))
    except Exception:
        return None
    return cls, nums


def pairs(seq: List[float]) -> List[Tuple[float, float]]:
    return list(zip(seq[0::2], seq[1::2]))


def poly_area_norm(pts: List[Tuple[float, float]]) -> float:
    n = len(pts)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def clamp01(nums: List[float]) -> List[float]:
    return [0.0 if v < 0.0 else 1.0 if v > 1.0 else v for v in nums]


def process_file(
    path: str,
    min_area: float,
    clamp: bool,
    max_per_image: Optional[int],
    dry_run: bool,
    backup: bool,
) -> Tuple[int, int]:
    """Returns (kept, dropped)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except Exception:
        return 0, 0

    out_lines: List[str] = []
    dropped = 0

    for ln in lines:
        parsed = parse_line(ln)
        if not parsed:
            dropped += 1
            continue
        cls, nums = parsed
        if len(nums) < 6 or len(nums) % 2 == 1:
            dropped += 1
            continue
        if clamp:
            nums = clamp01(nums)
        pts = pairs(nums)
        area = poly_area_norm(pts)
        if area < min_area:
            dropped += 1
            continue
        out_lines.append(f"{cls} " + " ".join(f"{v:.6f}" for v in nums))

    if max_per_image is not None and len(out_lines) > max_per_image:
        # Keep largest-area first; recompute areas for sorting
        def area_of(line: str) -> float:
            parts = line.split()
            nums = list(map(float, parts[1:]))
            return poly_area_norm(pairs(nums))

        out_lines.sort(key=area_of, reverse=True)
        dropped += len(out_lines) - max_per_image
        out_lines = out_lines[:max_per_image]

    kept = len(out_lines)
    if not dry_run:
        if backup and kept + dropped > 0:
            try:
                os.replace(path, path + ".bak")
            except Exception:
                pass
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(out_lines) + ("\n" if out_lines else ""))

    return kept, dropped


def main():
    ap = argparse.ArgumentParser(
        description="Prune tiny/malformed YOLO-seg polygons from labels")
    ap.add_argument("--root", required=True,
                    help="Dataset root containing labels/train and labels/val")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val"], help="Splits to process")
    ap.add_argument("--min-area", type=float, default=1e-4,
                    help="Minimum normalized polygon area to keep")
    ap.add_argument("--clamp", action="store_true",
                    help="Clamp coordinates to [0,1]")
    ap.add_argument("--max-per-image", type=int, default=None,
                    help="Max polygons per image (keep largest areas)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report without modifying files")
    ap.add_argument("--backup", action="store_true",
                    help="Backup original .txt to .bak before writing")
    args = ap.parse_args()

    total_files = 0
    total_kept = 0
    total_dropped = 0

    for sp in args.splits:
        files = list_label_files(args.root, sp)
        print(f"Split {sp}: {len(files)} label files")
        for p in files:
            kept, dropped = process_file(
                p,
                min_area=args.min_area,
                clamp=args.clamp,
                max_per_image=args.max_per_image,
                dry_run=args.dry_run,
                backup=args.backup,
            )
            total_files += 1
            total_kept += kept
            total_dropped += dropped

    print("-" * 60)
    print(f"Processed label files: {total_files}")
    print(f"Kept polygons: {total_kept}")
    print(f"Dropped polygons: {total_dropped}")
    if args.dry_run:
        print("No files modified (dry-run). Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
