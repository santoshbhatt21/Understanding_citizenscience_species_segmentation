#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yolo_seg_labels_cleaner.py
--------------------------
Clean YOLO **instance segmentation** labels created from dense masks (e.g., CAM+SAM) by:
  - merging micro-fragments per class (morphological closing + dilation),
  - discarding tiny components,
  - keeping only the top-K largest components per class,
  - simplifying polygons (Ramer–Douglas–Peucker),
  - clamping/validating normalized coords,
  - writing cleaned labels to a separate folder (no destructive overwrite),
  - optionally saving before/after overlays for QA.

This produces YOLO-friendly labels with **few, large, simple** instances per image,
which train much better than hundreds of tiny polygons.

USAGE (PowerShell examples)
---------------------------
# A) Dataset root with standard structure (images/<split>, labels/<split>)
python yolo_seg_labels_cleaner.py --root "E:/dataset" --splits train val

# B) Explicit directories
python yolo_seg_labels_cleaner.py --images-dir "E:/dataset/images/train" --labels-dir "E:/dataset/labels/train"

# Common knobs
  --min-area-frac 0.001   # drop polygons < 0.1% of image area
  --topk-per-class 3      # keep at most K components per class per image
  --close-k 7 --dilate-k 3
  --approx-eps-frac 0.003 # simplify polygon (epsilon = frac * perimeter)
  --max-vertices 200      # after simplify, cap vertices
  --save-overlays 20      # save up to N overlay PNGs for QA

OUTPUT
------
Writes cleaned labels under:
  - <root>/labels_clean/<split>/... (when using --root)
  - <labels-dir>/../labels_clean/... (when using --labels-dir)
Saves per-split stats JSON and optional overlays (before + after).

Notes
-----
- Polygons are expected in YOLO format per line: class x1 y1 x2 y2 ...
- All coords must be normalized [0,1]. This script clamps small numeric drift.
- If an image has no valid polygons after cleaning, an empty .txt is written.
"""
import os
import json
import math
import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp",
            ".tif", ".tiff", ".JPG", ".PNG", ".JPEG"}

# ------------------------------
# USER CONFIG (used when no CLI args are provided)
# ------------------------------
USER_CONFIG = {
    # Set this to your dataset root containing images/train, labels/train, images/val, labels/val
    "root": r"E:/Santosh_master_thesis/DATA_YOLO11_classified_Leaves_Trunks",
    "splits": ["train", "val"],
    # Cleaning knobs (same as CLI defaults)
    "min_area_frac": 0.001,
    "topk_per_class": 3,
    "close_k": 7,
    "dilate_k": 3,
    "approx_eps_frac": 0.003,
    "max_vertices": 200,
    # QA overlays: save up to N per split (0 disables)
    "save_overlays": 0,
}

# ------------------------------
# IO helpers
# ------------------------------


def _discover_pairs(images_dir: Path, labels_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for img in images_dir.rglob("*"):
        if img.suffix not in IMG_EXTS:
            continue
        rel = img.relative_to(images_dir)
        lbl = labels_dir / rel.with_suffix(".txt")
        pairs.append((img, lbl))
    return pairs


def _default_out_labels(labels_dir: Path) -> Path:
    parent = labels_dir.parent
    out = parent / "labels_clean" / labels_dir.name
    out.mkdir(parents=True, exist_ok=True)
    return out

# ------------------------------
# YOLO polygon helpers
# ------------------------------


def read_yolo_polys(label_path: Path) -> List[Tuple[int, np.ndarray]]:
    """Return list of (class_id, Nx2 normalized coords)."""
    polys = []
    if not label_path.exists():
        return polys
    try:
        for ln in label_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            parts = ln.strip().split()
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0 or len(coords) < 6:
                continue
            xs = np.asarray(coords[0::2], dtype=np.float32)
            ys = np.asarray(coords[1::2], dtype=np.float32)
            pts = np.stack([xs, ys], axis=1)  # Nx2 normalized
            polys.append((cls, pts))
    except Exception:
        pass
    return polys


def write_yolo_polys(label_path: Path, items: List[Tuple[int, np.ndarray]]) -> None:
    """items: list of (class_id, Nx2 pixel coords) — will be normalized using image size (stored separately)."""
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, pts_norm in items:
            if pts_norm.shape[0] < 3:
                continue
            coords = " ".join(f"{v:.6f}" for v in pts_norm.flatten().tolist())
            f.write(f"{cls} {coords}\n")


def clamp01(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)

# ------------------------------
# Rasterization & merging
# ------------------------------


def polygons_to_mask(polys: List[np.ndarray], w: int, h: int) -> np.ndarray:
    """Rasterize a list of normalized Nx2 polygons to a binary mask (uint8 {0,255})."""
    M = np.zeros((h, w), dtype=np.uint8)
    for P in polys:
        if P.shape[0] < 3:
            continue
        pts = (P * np.array([w, h], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(M, [pts], 255)
    return M


def mask_to_simplified_contours(mask: np.ndarray, approx_eps_frac: float, min_area_px: int, topk: int) -> List[np.ndarray]:
    """Return up to topk simplified contours (pixel coords Nx2) sorted by area desc."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area_px]
    if not contours:
        return []
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:topk]
    simple = []
    for c in contours:
        peri = float(cv2.arcLength(c, True))
        eps = approx_eps_frac * peri
        a = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)  # Nx2
        simple.append(a.astype(np.float32))
    return simple

# ------------------------------
# Overlays for QA
# ------------------------------


def overlay_polygons(img: Image.Image, polys_norm: List[Tuple[int, np.ndarray]], color=(0, 255, 0, 140)) -> Image.Image:
    W, H = img.size
    ov = img.copy()
    dr = ImageDraw.Draw(ov, "RGBA")
    for cls, P in polys_norm:
        pts = [(float(x*W), float(y*H)) for x, y in P]
        dr.polygon(pts, fill=color, outline=(
            color[0], color[1], color[2], 255))
    return ov

# ------------------------------
# Core cleaning per image
# ------------------------------


def clean_one_image(
    img_path: Path,
    lbl_path: Path,
    out_lbl_path: Path,
    min_area_frac: float = 0.001,
    topk_per_class: int = 3,
    close_k: int = 7,
    dilate_k: int = 3,
    approx_eps_frac: float = 0.003,
    max_vertices: int = 200,
) -> Dict:
    """Return stats dict; writes cleaned label file."""
    # read image size
    with Image.open(img_path) as im:
        W, H = im.size

    # read original polygons (normalized)
    polys = read_yolo_polys(lbl_path)
    n_orig = len(polys)
    total_pts_orig = int(sum(P.shape[0] for _, P in polys))

    # organize by class
    by_class: Dict[int, List[np.ndarray]] = {}
    for cls, P in polys:
        P = clamp01(P)  # clamp before rasterization to avoid OOB
        by_class.setdefault(cls, []).append(P)

    kept_items_norm: List[Tuple[int, np.ndarray]] = []
    min_area_px = max(1, int(min_area_frac * W * H))

    # process each class separately
    for cls, plist in by_class.items():
        # rasterize all polys of this class into one mask
        M = polygons_to_mask(plist, W, H)

        # morphology to merge small fragments
        if close_k > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_k, close_k))
            M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, k)
        if dilate_k > 0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
            M = cv2.dilate(M, k)

        # contours -> simplify -> filter by area -> keep top K
        contours_px = mask_to_simplified_contours(
            M, approx_eps_frac, min_area_px, topk_per_class)

        # optionally cap vertices and normalize
        for Cpx in contours_px:
            if Cpx.shape[0] > max_vertices:
                # simplify again by increasing epsilon slightly
                peri = float(cv2.arcLength(Cpx.reshape(-1, 1, 2), True))
                eps = 1.5 * approx_eps_frac * peri
                Cpx = cv2.approxPolyDP(
                    Cpx.reshape(-1, 1, 2), eps, True).reshape(-1, 2)
            # normalize
            Pn = np.zeros_like(Cpx, dtype=np.float32)
            Pn[:, 0] = Cpx[:, 0] / float(W)
            Pn[:, 1] = Cpx[:, 1] / float(H)
            Pn = clamp01(Pn)
            if Pn.shape[0] >= 3:
                kept_items_norm.append((cls, Pn))

    # write cleaned label
    write_yolo_polys(out_lbl_path, kept_items_norm)

    stats = {
        "image": str(img_path),
        "label_in": str(lbl_path),
        "label_out": str(out_lbl_path),
        "img_w": W, "img_h": H,
        "n_polygons_in": n_orig,
        "n_polygons_out": len(kept_items_norm),
        "total_pts_in": total_pts_orig,
        "total_pts_out": int(sum(P.shape[0] for _, P in kept_items_norm)),
    }
    return stats

# ------------------------------
# CLI
# ------------------------------


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Clean YOLO polygons for instance segmentation (merge fragments, drop tiny, simplify, top-K).")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--root", help="Dataset root with images/<split> and labels/<split>.")
    grp.add_argument(
        "--labels-dir", help="Explicit labels directory (use with --images-dir).")
    ap.add_argument(
        "--images-dir", help="Explicit images directory (required if --labels-dir is used).")

    ap.add_argument("--splits", nargs="+",
                    default=["train"], help="Splits under --root to process (default: train).")
    ap.add_argument("--out-labels-dir", default=None,
                    help="Where to write cleaned labels. Default is labels_clean/<split>.")

    # cleaning knobs
    ap.add_argument("--min-area-frac", type=float, default=0.001)
    ap.add_argument("--topk-per-class", type=int, default=3)
    ap.add_argument("--close-k", type=int, default=7)
    ap.add_argument("--dilate-k", type=int, default=3)
    ap.add_argument("--approx-eps-frac", type=float, default=0.003)
    ap.add_argument("--max-vertices", type=int, default=200)

    # QA
    ap.add_argument("--save-overlays", type=int, default=0,
                    help="Save up to N before/after overlays per split.")
    return ap.parse_args()


def _save_overlay(img_path: Path, polys_norm: List[Tuple[int, np.ndarray]], out_path: Path, color=(0, 255, 0, 140)):
    with Image.open(img_path).convert("RGB") as im:
        W, H = im.size
        ov = im.copy()
        dr = ImageDraw.Draw(ov, "RGBA")
        for cls, P in polys_norm:
            pts = [(float(x*W), float(y*H)) for x, y in P]
            dr.polygon(pts, fill=color, outline=(
                color[0], color[1], color[2], 255))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ov.save(out_path)


def main():
    # If no CLI args given, use USER_CONFIG
    if len(sys.argv) <= 1:
        class _NS:  # simple namespace
            pass
        args = _NS()
        args.root = USER_CONFIG.get("root")
        args.labels_dir = None
        args.images_dir = None
        args.splits = USER_CONFIG.get("splits", ["train"]) or ["train"]
        args.out_labels_dir = None
        args.min_area_frac = USER_CONFIG.get("min_area_frac", 0.001)
        args.topk_per_class = USER_CONFIG.get("topk_per_class", 3)
        args.close_k = USER_CONFIG.get("close_k", 7)
        args.dilate_k = USER_CONFIG.get("dilate_k", 3)
        args.approx_eps_frac = USER_CONFIG.get("approx_eps_frac", 0.003)
        args.max_vertices = USER_CONFIG.get("max_vertices", 200)
        args.save_overlays = USER_CONFIG.get("save_overlays", 0)
        if not args.root:
            raise SystemExit(
                "Please set USER_CONFIG['root'] to your dataset path.")
    else:
        args = _parse_args()

    if args.labels_dir and not args.images_dir:
        raise SystemExit("--images-dir is required when using --labels-dir.")

    if args.root:
        root = Path(args.root)
        for split in args.splits:
            im_dir = root / "images" / split
            lb_dir = root / "labels" / split
            if not im_dir.is_dir() or not lb_dir.is_dir():
                print(
                    f"[WARN] Missing {im_dir} or {lb_dir}, skipping split '{split}'.")
                continue
            pairs = _discover_pairs(im_dir, lb_dir)
            if not pairs:
                print(f"[WARN] No images under {im_dir}, skipping.")
                continue
            out_dir = Path(
                args.out_labels_dir) if args.out_labels_dir else _default_out_labels(lb_dir)

            stats = []
            # optional overlays (random subset)
            sel_idxs = set(random.sample(range(len(pairs)), k=min(
                args.save_overlays, len(pairs)))) if args.save_overlays > 0 else set()

            for i, (img_p, lbl_p) in enumerate(pairs):
                rel = img_p.relative_to(im_dir)
                out_lbl = out_dir / rel.with_suffix(".txt")
                s = clean_one_image(
                    img_p, lbl_p, out_lbl,
                    min_area_frac=args.min_area_frac,
                    topk_per_class=args.topk_per_class,
                    close_k=args.close_k,
                    dilate_k=args.dilate_k,
                    approx_eps_frac=args.approx_eps_frac,
                    max_vertices=args.max_vertices,
                )
                stats.append(s)

                # overlays
                if i in sel_idxs:
                    # before
                    polys_in = read_yolo_polys(lbl_p)
                    _save_overlay(img_p, polys_in, out_dir.parent / "qc_overlays" /
                                  split / (rel.stem + "_before.png"), color=(255, 0, 0, 120))
                    # after
                    polys_out = read_yolo_polys(out_lbl)
                    _save_overlay(img_p, polys_out, out_dir.parent / "qc_overlays" /
                                  split / (rel.stem + "_after.png"), color=(0, 255, 0, 120))

                if (i+1) % 100 == 0:
                    print(f"[{split}] {i+1}/{len(pairs)} processed...")

            # write stats
            stat_path = out_dir.parent / f"clean_stats_{split}.json"
            with open(stat_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            # brief summary
            n_in = sum(x["n_polygons_in"] for x in stats)
            n_out = sum(x["n_polygons_out"] for x in stats)
            print(
                f"[{split}] images={len(stats)}  polys_in={n_in} -> polys_out={n_out}")
            print(f"Cleaned labels: {out_dir}")
            if args.save_overlays:
                print(f"QA overlays: {out_dir.parent / 'qc_overlays' / split}")

    else:
        # explicit dirs
        im_dir = Path(args.images_dir)
        lb_dir = Path(args.labels_dir)
        pairs = _discover_pairs(im_dir, lb_dir)
        out_dir = Path(
            args.out_labels_dir) if args.out_labels_dir else _default_out_labels(lb_dir)
        stats = []
        sel_idxs = set(random.sample(range(len(pairs)), k=min(
            args.save_overlays, len(pairs)))) if args.save_overlays > 0 else set()

        for i, (img_p, lbl_p) in enumerate(pairs):
            rel = img_p.relative_to(im_dir)
            out_lbl = out_dir / rel.with_suffix(".txt")
            s = clean_one_image(
                img_p, lbl_p, out_lbl,
                min_area_frac=args.min_area_frac,
                topk_per_class=args.topk_per_class,
                close_k=args.close_k,
                dilate_k=args.dilate_k,
                approx_eps_frac=args.approx_eps_frac,
                max_vertices=args.max_vertices,
            )
            stats.append(s)
            if i in sel_idxs:
                polys_in = read_yolo_polys(lbl_p)
                _save_overlay(img_p, polys_in, out_dir.parent / "qc_overlays" /
                              (rel.stem + "_before.png"), color=(255, 0, 0, 120))
                polys_out = read_yolo_polys(out_lbl)
                _save_overlay(img_p, polys_out, out_dir.parent / "qc_overlays" /
                              (rel.stem + "_after.png"), color=(0, 255, 0, 120))

            if (i+1) % 100 == 0:
                print(f"[dir] {i+1}/{len(pairs)} processed...")

        stat_path = out_dir.parent / "clean_stats.json"
        with open(stat_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        n_in = sum(x["n_polygons_in"] for x in stats)
        n_out = sum(x["n_polygons_out"] for x in stats)
        print(
            f"[dir] images={len(stats)}  polys_in={n_in} -> polys_out={n_out}")
        print(f"Cleaned labels: {out_dir}")
        if args.save_overlays:
            print(f"QA overlays: {out_dir.parent / 'qc_overlays'}")


if __name__ == "__main__":
    main()
