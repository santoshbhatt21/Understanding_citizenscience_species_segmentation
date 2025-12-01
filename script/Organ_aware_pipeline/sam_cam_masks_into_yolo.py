#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM/any binary trunk masks → YOLOv8/v11 Segmentation labels (TRUNKS ONLY)

Simplified for *trunks* (no broadleaf vs conifer logic). Key traits:
- Treats any non-background mask as **trunk foreground** (binary).
- Normalizes polygon coordinates by the *image* size (W,H).
- Uses RETR_EXTERNAL + CHAIN_APPROX_TC89_L1 for smooth, stable contours.
- Speckle removal + explicit min/max **area gates** (relative to image area).
- Gentle polygon simplification (epsilon as small fraction of perimeter).
- Instance selection: keep the **largest** component by default (typical for a single trunk),
  or choose keep_all / top-K.
- Optional overlays for quick visual sanity checks.
- Stable class-name ↔ id mapping (classes.json) inferred from folder names,
  or pass a fixed --class-id.

Typical usage (per-class folders):
    python Sam_Mask_to_YOLO_trunks.py \
        --images E:/.../classified_Trunks \
        --masks  E:/.../classified_Trunks \
        --labels E:/.../DATA_YOLO11_Trunks/labels \
        --merge-mode largest \
        --min-area-frac 0.001 --max-area-frac 0.80 \
        --epsilon-frac 0.0012 \
        --labels-per-class \
        --save-overlays 50

Assumptions:
- Masks share the same base filename as images (basename match). Extensions may differ.
- If mask size != image size, mask is resized to image size (INTER_NEAREST).
- Writes YOLO-seg labels per image: one line per instance
    class_id x1 y1 x2 y2 ... xN yN
- For trunk-only binary masks: any mask > 0 is treated as foreground.
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
try:
    import yaml  # optional, only used if config file is YAML
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False

# ----------------------------- helpers ---------------------------------
IMG_EXTS_DEFAULT = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
MASK_EXTS_DEFAULT = [".png", ".bmp", ".tif", ".tiff"]

# Common filename decorations often seen on mask files
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
    """Yield candidate basenames for matching an image given a mask basename.
    Strategy:
    - Try exact raw name first
    - Try stripping any known prefix
    - Try stripping any known suffix (with optional trailing digits)
    - Try stripping both a prefix and a suffix.
    Duplicates removed, order preserved.
    """
    cands: List[str] = []

    def add(x: str):
        if x not in cands and len(x) > 0:
            cands.append(x)

    add(raw)

    # strip prefixes
    for pref in _MASK_PREFIXES:
        if raw.lower().startswith(pref):
            add(raw[len(pref):])

    # strip suffixes (and suffix + up to 4 trailing zeros/digits)
    lower = raw.lower()
    for suf in _MASK_SUFFIXES:
        if lower.endswith(suf):
            add(raw[: -len(suf)])
        for k in range(1, 5):
            tail = f"{suf}{'0'*k}"
            if lower.endswith(tail):
                add(raw[: -len(tail)])

    # strip both prefix and suffix combinations
    for pref in _MASK_PREFIXES:
        if raw.lower().startswith(pref):
            mid = raw[len(pref):]
            midl = mid.lower()
            for suf in _MASK_SUFFIXES:
                if midl.endswith(suf):
                    add(mid[: -len(suf)])
                for k in range(1, 5):
                    tail = f"{suf}{'0'*k}"
                    if midl.endswith(tail):
                        add(mid[: -len(tail)])

    return cands


def index_images(images_root: Path, img_exts: Sequence[str]) -> Tuple[Dict[str, Path], Dict[str, int]]:
    """Map basename (no ext) → image path. Count duplicates.
    Keeps first occurrence on collisions.
    """
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
    """Any non-zero becomes foreground. Returns cleaned binary mask in {0,255}."""
    m = (mask > 0).astype(np.uint8) * 255
    h, w = m.shape[:2]
    img_area = float(h * w)

    if do_morph:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # remove speckle, then gently seal small gaps (avoid over-closing for trunks)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    # drop tiny components
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
    # infer from immediate parent folder name
    cls_name = img_path.parent.name
    if cls_name not in classes_map:
        classes_map[cls_name] = len(classes_map)
    return classes_map[cls_name], classes_map

# ------------------------------ main logic -------------------------------


def process(args) -> None:
    images_root = Path(args.images).resolve()
    masks_root = Path(args.masks).resolve()
    labels_root = Path(args.labels).resolve()
    ensure_dir(labels_root)

    img_exts = [e if e.startswith(
        ".") else "." + e for e in (args.img_exts or IMG_EXTS_DEFAULT)]
    mask_exts = [e if e.startswith(
        ".") else "." + e for e in (args.mask_exts or MASK_EXTS_DEFAULT)]

    classes_map_path = labels_root / (args.class_map or "classes.json")
    classes_map = load_or_init_classes(classes_map_path)

    mask_files = list_files(masks_root, mask_exts)
    if not mask_files:
        raise RuntimeError(
            f"No mask files found under {masks_root} with extensions {mask_exts}")

    random.seed(0)
    overlay_dir = labels_root / "_overlays" if args.save_overlays > 0 else None
    if overlay_dir:
        ensure_dir(overlay_dir)

    kept = 0
    skipped = 0
    skipped_unmatched = 0
    skipped_no_contour = 0
    overlay_left = args.save_overlays

    # Pre-index images for speed
    image_index, dup_counts = index_images(images_root, img_exts)
    if dup_counts:
        print(
            f"[WARN] Duplicate image basenames detected: {len(dup_counts)} (only first occurrence used)")
    print(
        f"[FOUND] masks: {len(mask_files)} | images index: {len(image_index)}")

    for mpath in mask_files:
        base_raw = basename_no_ext(mpath)
        ipath: Optional[Path] = None
        matched_base: Optional[str] = None
        # Try exact, then heuristic candidates
        for cand in candidate_basenames(base_raw):
            ipath = image_index.get(cand)
            if ipath is not None:
                matched_base = cand
                break
        if ipath is None:
            print(f"[WARN] No matching image for mask (by basename): {mpath}")
            skipped += 1
            skipped_unmatched += 1
            continue

        # Read image & mask, enforce size match
        im = imread_color(ipath)
        H, W = im.shape[:2]
        mask = imread_gray(mpath)
        if mask.shape[:2] != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        # Optional threshold before inversion/morphology
        # --- Threshold to binary first (0/255) ---
        if args.mask_threshold is not None:
            thr = int(np.clip(args.mask_threshold, 0, 255))
            mask = (mask >= thr).astype(np.uint8) * 255

        # --- Polarity handling (compute fg_ratio ONCE) ---
        # 'fg_white' | 'fg_black' | 'auto'
        pol = getattr(args, "polarity", "fg_white")
        # fraction of nonzero pixels
        fg_ratio = float((mask > 0).mean())

        if pol == "fg_black":
            # Common case: trunk is black (0) -> flip so FG becomes 255
            mask = 255 - mask
        elif pol == "auto":
            if fg_ratio < args.auto_invert_low or fg_ratio > (1.0 - args.auto_invert_low):
                mask = 255 - mask
        # else: fg_white -> do nothing

        # --- (Optional) Legacy flags still honored, but not required ---
        if getattr(args, "invert_masks", False) and pol != "auto":
            mask = 255 - mask
        if getattr(args, "auto_invert", False) and pol != "auto":
            # recompute after potential flips
            fg_ratio = float((mask > 0).mean())
            if fg_ratio < args.auto_invert_low or fg_ratio > (1.0 - args.auto_invert_low):
                mask = 255 - mask

        # Clean & contour
        mask_clean = clean_binary_mask(
            mask, min_area_frac=args.min_area_frac, do_morph=not args.no_morphology)
        cnts = find_contours(mask_clean)
        cnts = filter_contours_by_area(
            cnts, W, H, args.min_area_frac, args.max_area_frac)
        if not cnts:
            # write empty label (optional) – use IMAGE basename for YOLO alignment
            base_out = ipath.stem
            (labels_root / f"{base_out}.txt").write_text("", encoding="utf-8")
            skipped += 1
            skipped_no_contour += 1
            continue

        # Select instances (largest by default for single-trunk use-case)
        cnts = sort_contours_by_area(cnts, reverse=True)
        mode = args.merge_mode.lower()
        if mode == "largest":
            cnts = cnts[:max(1, args.topk)]
        elif mode == "keep_all":
            if args.topk > 0:
                cnts = cnts[:args.topk]
        elif mode == "hull":
            pts = np.vstack([c.reshape(-1, 2) for c in cnts])
            hull = cv2.convexHull(pts.astype(np.float32), clockwise=False)
            cnts = [hull]
        else:
            print(
                f"[WARN] Unknown merge_mode '{args.merge_mode}', defaulting to 'largest'")
            cnts = cnts[:1]

        # Polygons (normalized) with a single epsilon for trunks
        polys = contours_to_polys_norm(cnts, W, H, eps_frac=args.epsilon_frac)
        polys = [p for p in polys if len(p) >= 3]

        # Determine class id
        class_id, classes_map = get_class_for_image(
            ipath, args.class_id, classes_map)
        class_name = ipath.parent.name if args.class_id is None else f"class_{class_id}"

        # Write YOLO seg label file (one line per instance) – name after IMAGE basename
        base_out = ipath.stem
        if args.labels_per_class:
            class_dir = labels_root / class_name
            ensure_dir(class_dir)
            lpath = class_dir / f"{base_out}.txt"
        else:
            lpath = labels_root / f"{base_out}.txt"
        with open(lpath, "w", encoding="utf-8") as f:
            for poly in polys:
                flat = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
                f.write(f"{class_id} {flat}\n")

        kept += 1

        # Optional overlay
        if overlay_dir and overlay_left > 0:
            polys_px = [np.column_stack(
                [p[:, 0] * W, p[:, 1] * H]) for p in polys]
            vis = overlay_polys(im, polys_px, color=(0, 255, 0))
            cv2.imwrite(str(overlay_dir / f"{base_out}_overlay.jpg"), vis)
            overlay_left -= 1

    # Save class mapping
    save_classes(classes_map_path, classes_map)

    print(f"[DONE] Wrote labels: {kept}  |  skipped total: {skipped}")
    if skipped_unmatched or skipped_no_contour:
        print(
            f"[SKIP REASONS] unmatched: {skipped_unmatched} | no-contour/area-gate: {skipped_no_contour}")
    print(f"[CLASSES] -> {classes_map_path}  ({len(classes_map)} classes)")

# ------------------------------- args ------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert trunk masks to YOLOv8/v11-seg labels (cleaned, trunks only).")
    # Optional config file (JSON or YAML) to supply defaults
    p.add_argument("--config", type=str, default=None,
                   help="Optional JSON/YAML config path.")

    # Required I/O
    p.add_argument("--images", required=True,
                   help="Root dir of images (class subfolders ok).")
    p.add_argument("--masks", required=True,
                   help="Root dir of trunk masks (same basename as images).")
    p.add_argument("--labels", required=True,
                   help="Output dir for YOLO seg label .txt files.")

    # Class handling
    p.add_argument("--class-id", type=int, default=None,
                   help="Explicit class id (if images not in class subfolders).")
    p.add_argument("--class-map", type=str, default="classes.json",
                   help="Filename for persistent class-name↔id map inside labels dir.")

    # Instance selection
    p.add_argument("--merge-mode", type=str, default="largest", choices=[
                   "largest", "keep_all", "hull"], help="How to select instances from mask components.")
    p.add_argument("--topk", type=int, default=1,
                   help="If keep_all/largest, keep top-K by area (default 1).")

    # File types
    p.add_argument("--img-exts", nargs="+", default=IMG_EXTS_DEFAULT,
                   help="Image extensions to search.")
    p.add_argument("--mask-exts", nargs="+",
                   default=MASK_EXTS_DEFAULT, help="Mask extensions to include.")

    # Geometry & cleaning
    p.add_argument("--min-area-frac", type=float, default=0.001,
                   help="Drop components smaller than this fraction of image area (e.g., 0.001=0.1%).")
    p.add_argument("--max-area-frac", type=float, default=0.80,
                   help="Drop components larger than this fraction of image area.")
    p.add_argument("--epsilon-frac", type=float, default=0.0012,
                   help="Polygon simplification epsilon fraction (perimeter-relative).")

    # Mask binarization & morphology
    p.add_argument("--mask-threshold", type=int, default=None,
                   help="Optional fixed threshold (0-255) before cleaning; >= value becomes foreground.")
    p.add_argument("--invert-masks", action="store_true",
                   help="Explicitly invert mask (if FG is black on white).")
    p.add_argument("--auto-invert", action="store_true",
                   help="Heuristic inversion for extreme FG ratios.")
    p.add_argument("--auto-invert-low", type=float, default=0.01,
                   help="Lower FG ratio threshold for auto-invert.")
    p.add_argument("--no-morphology", action="store_true",
                   help="Disable morphological open/close (keep raw mask).")

    # Output layout
    p.add_argument("--labels-per-class", action="store_true",
                   help="Write labels in subfolders named after class.")
    p.add_argument("--save-overlays", type=int, default=0,
                   help="Save this many overlay JPGs to labels/_overlays for sanity checks.")
    # Polarity of trunk masks
    p.add_argument(
        "--polarity", type=str, default="fg_black",
        choices=["fg_white", "fg_black", "auto"],
        help="Mask polarity: 'fg_white' (object=white), 'fg_black' (object=black), or 'auto' (guess by FG ratio)."
    )
    return p


if __name__ == "__main__":
    # 1) Pre-parse to fetch --config without other required args
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_known, remaining = pre.parse_known_args(sys.argv[1:])

    # 2) Build the full parser
    parser = build_argparser()

    # 3) Load config defaults if provided
    cfg_path = pre_known.config or os.environ.get("TRUNK_MASK_TO_YOLO_CONFIG")
    if cfg_path:
        cfg_file = Path(cfg_path)
        if not cfg_file.exists():
            print(f"[WARN] Config file not found: {cfg_file}")
        else:
            try:
                cfg_data: Dict = {}
                if cfg_file.suffix.lower() in {".yml", ".yaml"}:
                    if not _HAS_YAML:
                        raise RuntimeError(
                            "YAML config provided but PyYAML is not installed.")
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        cfg_data = yaml.safe_load(f) or {}
                else:
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        cfg_data = json.load(f)
                if not isinstance(cfg_data, dict):
                    raise ValueError("Config root must be a key/value map.")
                known = {a.dest for a in parser._actions if a.dest not in {"help"}}
                defaults = {k: v for k, v in cfg_data.items() if k in known}
                if defaults:
                    parser.set_defaults(**defaults)
                    print(
                        f"[CONFIG] Loaded defaults from {cfg_file}: {sorted(defaults.keys())}")
            except Exception as e:
                print(f"[WARN] Failed to load config '{cfg_file}': {e}")

    # 4) Parse and run
    args = parser.parse_args(remaining)
    process(args)