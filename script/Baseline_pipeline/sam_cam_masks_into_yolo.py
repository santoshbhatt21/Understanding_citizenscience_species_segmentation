#!/usr/bin/env python
# -*- coding: utf-8 -*- 
"""
SAM/any binary leaf masks → YOLOv8/v11 Segmentation labels (LEAVES ONLY)

Key traits:
- Treats any non-background mask as **leaf foreground** (binary).
- Normalizes polygon coordinates by the *image* size (W,H).
- Uses RETR_EXTERNAL + CHAIN_APPROX_TC89_L1 for smooth, stable contours.
- Speckle removal + explicit min/max **area gates** (relative to image area).
- Gentle polygon simplification (epsilon as small fraction of perimeter).
- Instance selection: keep **all instances** for leaves (multiple leaves per image).
- Optional overlays for quick visual sanity checks.
- Stable class-name ↔ id mapping (classes.json) inferred from folder names,
  or pass a fixed --class-id.

Typical usage (per-class folders):
    python Sam_Mask_to_YOLO_leaves.py \
        --images E:/.../classified_Leaves \
        --masks  E:/.../classified_Leaves \
        --labels E:/.../DATA_YOLO11_Leaves/labels \
        --merge-mode keep_all \
        --min-area-frac 0.001 --max-area-frac 0.80 \
        --epsilon-frac 0.0012 \
        --labels-per-class \
        --save-overlays 50

Assumptions:
- Masks share the same base filename as images (basename match). Extensions may differ.
- If mask size != image size, mask is resized to image size (INTER_NEAREST).
- Writes YOLO-seg labels per image: one line per instance
    class_id x1 y1 x2 y2 ... xN yN
- For leaf-only binary masks: any mask > 0 is treated as foreground.
"""

import argparse
import json
import os
import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ----------------------------- helpers ---------------------------------
IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS_DEFAULT = [".png", ".bmp", ".tif", ".tiff"]

_MASK_PREFIXES = [
    "mask_", "msk_", "seg_", "m_", "lbl_", "label_", "binary_", "bin_", "pred_", "sam_", "cam_",
]
_MASK_SUFFIXES = [
    "_mask", "_msk", "_seg", "_labels", "_label", "-mask", "-msk", "-seg", "_binary", "_pred", "_sam", "_cam",
]

def imread_color(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return im

def imread_gray(path: Path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return im

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def list_files(root: Path, exts: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    exts = [e.lower() for e in exts]
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out

def basename_no_ext(p: Path) -> str:
    return p.stem

def candidate_basenames(raw: str) -> List[str]:
    cands: List[str] = []
    def add(x: str):
        if x not in cands and len(x) > 0:
            cands.append(x)

    add(raw)
    for pref in _MASK_PREFIXES:
        if raw.lower().startswith(pref):
            add(raw[len(pref):])

    for suf in _MASK_SUFFIXES:
        if raw.lower().endswith(suf):
            add(raw[: -len(suf)])
        for k in range(1, 5):
            tail = f"{suf}{'0'*k}"
            if raw.lower().endswith(tail):
                add(raw[: -len(tail)])

    return cands

def index_images(images_root: Path, img_exts: Sequence[str]) -> Tuple[Dict[str, Path], Dict[str, int]]:
    idx: Dict[str, Path] = {}
    dups: Dict[str, int] = {}
    exts_set = {e.lower() for e in img_exts}
    for p in images_root.rglob('*'):
        if p.is_file() and p.suffix.lower() in exts_set:
            b = p.stem
            if b in idx:
                dups[b] = dups.get(b, 1) + 1
            else:
                idx[b] = p
    return idx, dups

def overlay_polys(im: np.ndarray, polys_px: List[np.ndarray], color=(0, 255, 0)) -> np.ndarray:
    out = im.copy()
    for pts in polys_px:
        cv2.polylines(out, [pts.astype(np.int32)], True, color, 2)
    return out

# -------------------------- mask postprocessing --------------------------

def clean_binary_mask(mask: np.ndarray, min_area_frac: float = 0.001, do_morph: bool = True) -> np.ndarray:
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    img_area = float(h * w)

    if do_morph:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (m > 0).astype(np.uint8), 8)
    keep = np.zeros_like(m)
    min_area = max(1.0, min_area_frac * img_area)
    for i in range(1, num):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == i] = 255
    return keep

def find_contours(mask_255: np.ndarray) -> List[np.ndarray]:
    cnts, _ = cv2.findContours(
        mask_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    return cnts

def simplify_contour(cnt: np.ndarray, eps_frac: float) -> np.ndarray:
    peri = cv2.arcLength(cnt, True)
    eps = max(1e-6, eps_frac * peri)
    simp = cv2.approxPolyDP(cnt, eps, True)
    return simp if len(simp) >= 3 else cnt

def filter_contours_by_area(cnts: List[np.ndarray], w: int, h: int, min_area_frac: float, max_area_frac: float) -> List[np.ndarray]:
    img_area = float(w * h)
    out: List[np.ndarray] = []
    min_a = min_area_frac * img_area
    max_a = max_area_frac * img_area
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < min_a or a > max_a:
            continue
        out.append(c)
    return out

def sort_contours_by_area(cnts: List[np.ndarray], reverse: bool = True) -> List[np.ndarray]:
    return sorted(cnts, key=cv2.contourArea, reverse=reverse)

def contours_to_polys_norm(cnts: List[np.ndarray], w: int, h: int, eps_frac: float) -> List[np.ndarray]:
    polys: List[np.ndarray] = []
    for c in cnts:
        c = simplify_contour(c, eps_frac=eps_frac)
        pts = c.reshape(-1, 2).astype(np.float32)
        if len(pts) < 3:
            continue
        pts[:, 0] = np.clip(pts[:, 0] / float(w), 0.0, 1.0)
        pts[:, 1] = np.clip(pts[:, 1] / float(h), 0.0, 1.0)
        polys.append(pts)
    return polys

# ------------------------- class mapping utils ---------------------------

def load_or_init_classes(map_path: Path) -> Dict[str, int]:
    if map_path.exists():
        with open(map_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_classes(map_path: Path, mapping: Dict[str, int]) -> None:
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

def get_class_for_image(img_path: Path, explicit_id: Optional[int], classes_map: Dict[str, int]) -> Tuple[int, Dict[str, int]]:
    if explicit_id is not None:
        return int(explicit_id), classes_map
    cls_name = img_path.parent.name
    if cls_name not in classes_map:
        classes_map[cls_name] = len(classes_map)
    return classes_map[cls_name], classes_map

def _resolve_under_root(path_str: str, root: Optional[Path] = None) -> Path:
    """
    If --root is provided, resolve 'path_str' relative to it (when not absolute),
    otherwise just return the absolute path of 'path_str'.
    """
    p = Path(path_str)
    if not p.is_absolute() and root is not None:
        p = root / p
    return p.resolve()

# ------------------------------ main logic -------------------------------

def process(args) -> None:
    root = Path(args.root).resolve() if getattr(args, "root", None) else None
    images_root = _resolve_under_root(args.images)
    masks_root = _resolve_under_root(args.masks)
    labels_root = _resolve_under_root(args.labels)
    print(f"[PATHS] images={images_root} | masks={masks_root} | labels={labels_root}")
    ensure_dir(labels_root)

    img_exts = [e if e.startswith(".") else "." + e for e in (args.img_exts or IMG_EXTS_DEFAULT)]
    mask_exts = [e if e.startswith(".") else "." + e for e in (args.mask_exts or MASK_EXTS_DEFAULT)]

    classes_map_path = labels_root / (args.class_map or "classes.json")
    classes_map = load_or_init_classes(classes_map_path)

    mask_files = list_files(masks_root, mask_exts)
    if not mask_files:
        raise RuntimeError(f"No mask files found under {masks_root} with extensions {mask_exts}")

    random.seed(0)
    overlay_dir = labels_root / "_overlays" if args.save_overlays > 0 else None
    if overlay_dir:
        ensure_dir(overlay_dir)

    kept = 0
    skipped = 0

    # Pre-index images for speed
    image_index, dup_counts = index_images(images_root, img_exts)
    if dup_counts:
        print(f"[WARN] Duplicate image basenames detected: {len(dup_counts)} (only first occurrence used)")
    print(f"[FOUND] masks: {len(mask_files)} | images index: {len(image_index)}")

    for mpath in mask_files:
        base_raw = basename_no_ext(mpath)
        ipath: Optional[Path] = None
        matched_base: Optional[str] = None
        for cand in candidate_basenames(base_raw):
            ipath = image_index.get(cand)
            if ipath is not None:
                matched_base = cand
                break
        if ipath is None:
            print(f"[WARN] No matching image for mask (by basename): {mpath}")
            skipped += 1
            continue

        im = imread_color(ipath)
        H, W = im.shape[:2]
        mask = imread_gray(mpath)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        mask_clean = clean_binary_mask(mask, min_area_frac=args.min_area_frac)
        cnts = find_contours(mask_clean)
        cnts = filter_contours_by_area(cnts, W, H, args.min_area_frac, args.max_area_frac)
        if not cnts:
            base_out = ipath.stem
            (labels_root / f"{base_out}.txt").write_text("", encoding="utf-8")
            skipped += 1
            continue

        cnts = sort_contours_by_area(cnts, reverse=True)
        polys = contours_to_polys_norm(cnts, W, H, eps_frac=args.epsilon_frac)

        class_id, classes_map = get_class_for_image(ipath, args.class_id, classes_map)

        base_out = ipath.stem
        lpath = labels_root / f"{base_out}.txt"
        with open(lpath, "w", encoding="utf-8") as f:
            for poly in polys:
                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                f.write(f"{class_id} {flat}\n")

        kept += 1
        if overlay_dir:
            polys_px = [np.column_stack([p[:, 0] * W, p[:, 1] * H]) for p in polys]
            vis = overlay_polys(im, polys_px, color=(0, 255, 0))
            cv2.imwrite(str(overlay_dir / f"{base_out}_overlay.jpg"), vis)

    save_classes(classes_map_path, classes_map)

    print(f"[DONE] Wrote labels: {kept}  |  skipped total: {skipped}")

# ------------------------------- args ------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert leaf masks to YOLOv8/v11-seg labels (cleaned, leaves only).")
    p.add_argument("--config", type=str, default=None, help="Optional config path.")
    p.add_argument("--root", type=str, default=None, help="Optional root directory for relative paths.")
    p.add_argument("--images", required=True, help="Root dir of images (class subfolders ok).")
    p.add_argument("--masks", required=True, help="Root dir of leaf masks.")
    p.add_argument("--labels", required=True, help="Output dir for YOLO-seg label .txt files.")
    p.add_argument("--class-id", type=int, default=None, help="Explicit class id for leaves.")
    p.add_argument("--class-map", type=str, default="classes.json", help="Class ID mapping file.")
    p.add_argument("--merge-mode", type=str, default="keep_all", choices=["largest", "keep_all", "hull"], help="How to select instances.")
    p.add_argument("--topk", type=int, default=1, help="If keep_all/largest, keep top-K.")
    p.add_argument("--img-exts", nargs="+", default=IMG_EXTS_DEFAULT, help="Image extensions.")
    p.add_argument("--mask-exts", nargs="+", default=MASK_EXTS_DEFAULT, help="Mask extensions.")
    p.add_argument("--min-area-frac", type=float, default=0.001, help="Drop components smaller than this fraction of image area.")
    p.add_argument("--max-area-frac", type=float, default=0.80, help="Drop components larger than this fraction of image area.")
    p.add_argument("--epsilon-frac", type=float, default=0.0012, help="Polygon simplification epsilon.")
    p.add_argument("--mask-threshold", type=int, default=None, help="Fixed threshold for binary mask.")
    p.add_argument("--invert-masks", action="store_true", help="Invert mask polarity.")
    p.add_argument("--save-overlays", type=int, default=0, help="Save this many overlay images for checks.")
    return p

if __name__ == "__main__":
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre.add_argument("--root", type=str, default=None)
    pre_known, remaining = pre.parse_known_args(sys.argv[1:])
    parser = build_argparser()
    args = parser.parse_args(remaining)
    process(args)
