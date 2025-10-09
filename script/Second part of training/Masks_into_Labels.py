#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SAM masks → YOLO segmentation labels (with safe renaming of 'mask_' files)

- Mirrors your input folder structure into an output labels folder.
- Renames mask files in-place by removing the 'mask_' prefix (collision-safe).
- For each mask image:
    * extracts external contours,
    * simplifies polygons,
    * writes YOLOv8/YOLO11 segmentation labels: "class x1 y1 x2 y2 ...".
- Optional CLASS_MAP lets you set class IDs from folder names; otherwise class=0.

Tested with: Python 3.8+, OpenCV 4+.

Usage:
  1) Edit CONFIG below.
  2) Run: python sam_masks_to_yolo.py
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np

# ========= CONFIG (EDIT ME) =========
CONFIG = {
    # Root folder containing your SAM masks. Subfolders (e.g., train/val/classA/…) are preserved.
    "INPUT_MASK_ROOT": r"E:/Santosh_master_thesis/Mask_Leaves",

    # Where to write YOLO label .txt files (mirrors subfolders).
    "OUTPUT_LABELS_ROOT": r"E:/Santosh_master_thesis/labels_Leaves",

    # File patterns to treat as masks:
    "MASK_EXTENSIONS": [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],

    # Remove this prefix from filenames (rename on disk). Set to "" to disable.
    "REMOVE_PREFIX": "mask_",

    # If you want class IDs from folder names, put them here (case-sensitive).
    # The first matching key found in any ancestor folder (relative to INPUT_MASK_ROOT) is used.
    # Example: {"oak": 0, "pine": 1, "maple": 2}
    "CLASS_MAP": {},  # empty => all objects use class 0

    # Minimum contour area in pixels to keep (filters speckles).
    "MIN_AREA_PX": 50,

    # Polygon simplification strength (fraction of contour perimeter). Typical: 0.005–0.02
    "CONTOUR_EPS_FRAC": 0.01,

    # If True, write an empty .txt when nothing is detected; if False, skip writing.
    "SAVE_EMPTY_LABELS": True,

    # Dry-run: log what would happen without changing files.
    "DRY_RUN": False,

    # Overwrite existing label files if present
    "OVERWRITE_LABELS": True,
}
# ====================================


logger = logging.getLogger("sam2yolo")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def is_mask_file(p: Path, exts: List[str]) -> bool:
    return p.is_file() and p.suffix.lower() in [e.lower() for e in exts]


def strip_prefix(name: str, prefix: str) -> str:
    return name[len(prefix):] if prefix and name.startswith(prefix) else name


def find_class_id(rel_parts: List[str], class_map: Dict[str, int]) -> int:
    """Search from deepest to shallowest folder name for a CLASS_MAP hit."""
    if not class_map:
        return 0
    for part in reversed(rel_parts):
        if part in class_map:
            return class_map[part]
    return 0


def ensure_dir(path: Path, dry: bool):
    if not path.exists():
        if dry:
            logger.info(f"[DRY] mkdir -p {path}")
        else:
            path.mkdir(parents=True, exist_ok=True)


def read_mask_as_binary(mask_path: Path) -> np.ndarray:
    """Load mask (any format) and return binary uint8 (0/1)."""
    img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read: {mask_path}")
    # If multi-channel, convert to grayscale
    if img.ndim == 3:
        # If has alpha, prioritize it; else use grayscale
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            gray = alpha
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # Any non-zero pixel is foreground
    binary = (gray > 0).astype(np.uint8)
    return binary


def contours_to_yolo_lines(binary: np.ndarray,
                           class_id: int,
                           min_area_px: int,
                           eps_frac: float) -> List[str]:
    """Extract external contours → YOLO seg lines."""
    h, w = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_px:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = eps_frac * peri
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)

        # Normalize to [0,1] in (x,y) order as required by YOLO seg
        xs = np.clip(approx[:, 0] / float(w), 0, 1)
        ys = np.clip(approx[:, 1] / float(h), 0, 1)

        # YOLO wants: class x1 y1 x2 y2 ...
        coords = np.column_stack([xs, ys]).reshape(-1)
        line = f"{class_id} " + " ".join(f"{v:.6f}" for v in coords)
        lines.append(line)
    return lines


def write_label_file(label_path: Path, lines: List[str], save_empty: bool, overwrite: bool, dry: bool):
    if not lines and not save_empty:
        return
    if label_path.exists() and not overwrite:
        logger.warning(f"Label exists, skipping: {label_path}")
        return
    text = "\n".join(lines) if lines else ""
    if dry:
        logger.info(f"[DRY] write {label_path} ({len(lines)} objects)")
        return
    label_path.write_text(text, encoding="utf-8")


def safe_rename(src: Path, dst: Path, dry: bool) -> Path:
    """Rename src→dst unless dst exists. If it exists, keep src and warn."""
    if src == dst or dst.name == src.name:
        return src
    if dst.exists():
        logger.warning(f"Target exists, not renaming: {dst}. Keeping {src.name}.")
        return src
    if dry:
        logger.info(f"[DRY] rename {src} -> {dst.name}")
        return dst
    src.rename(dst)
    return dst


def process_all():
    inp_root = Path(CONFIG["INPUT_MASK_ROOT"]).resolve()
    out_root = Path(CONFIG["OUTPUT_LABELS_ROOT"]).resolve()
    exts = CONFIG["MASK_EXTENSIONS"]
    prefix = CONFIG["REMOVE_PREFIX"]
    class_map = CONFIG["CLASS_MAP"]
    min_area = int(CONFIG["MIN_AREA_PX"])
    eps_frac = float(CONFIG["CONTOUR_EPS_FRAC"])
    dry = bool(CONFIG["DRY_RUN"])
    save_empty = bool(CONFIG["SAVE_EMPTY_LABELS"])
    overwrite = bool(CONFIG["OVERWRITE_LABELS"])

    if not inp_root.exists():
        logger.error(f"INPUT_MASK_ROOT does not exist: {inp_root}")
        sys.exit(1)

    logger.info(f"Input masks : {inp_root}")
    logger.info(f"Output labels: {out_root}")
    logger.info(f"Remove prefix: '{prefix}'")
    logger.info(f"Dry-run      : {dry}")

    files = [p for p in inp_root.rglob("*") if is_mask_file(p, exts)]
    if not files:
        logger.warning("No mask files found.")
        return

    for mask_path in files:
        rel = mask_path.relative_to(inp_root)
        rel_dir = rel.parent  # subfolder path to mirror
        # Choose class id
        class_id = find_class_id(list(rel_dir.parts), class_map)

        # Determine new filename (strip prefix)
        new_name = strip_prefix(mask_path.name, prefix)
        new_mask_path = mask_path.with_name(new_name)

        # Actually rename on disk (collision-safe)
        final_mask_path = safe_rename(mask_path, new_mask_path, dry=dry)

        # Compute destination label path (mirror folders) using final name (without prefix)
        label_dir = out_root.joinpath(rel_dir)
        ensure_dir(label_dir, dry=dry)
        label_path = label_dir.joinpath(final_mask_path.stem + ".txt")

        # Read binary mask and convert to YOLO lines
        try:
            binary = read_mask_as_binary(final_mask_path if final_mask_path.exists() else mask_path)
            lines = contours_to_yolo_lines(binary, class_id, min_area, eps_frac)
        except Exception as e:
            logger.error(f"Failed on {final_mask_path.name}: {e}")
            continue

        write_label_file(label_path, lines, save_empty, overwrite, dry)

    logger.info("Done.")


if __name__ == "__main__":
    process_all()
