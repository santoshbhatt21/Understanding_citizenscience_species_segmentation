#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class panels and collages for mask QA.

Upgrades:
- Optional mask cleaning (morph open/close + area gates) for clearer overlays.
- Auto-invert polarity by foreground ratio.
- Overlay modes: 'edge' (contours) or 'fill' (tint FG).
- QC: skip near-empty or near-full masks.
- Layouts: original 3-column panel or a compact overlay-only collage across classes (e.g., 10 classes in one PNG).
"""

import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

# -------------------- USER SETTINGS --------------------
IMAGES_ROOT = r"E:/Santosh_master_thesis/Classified_Leaves"     # classes as subfolders
MASKS_ROOT = r"E:/Santosh_master_thesis/Classified_Masks_binary"      # same class subfolders
OUTPUT_PATH = r"E:/Santosh_master_thesis/image_for_writing/collage.png"

SAMPLES_PER_CLASS = 1
TILE_SIZE = 320
RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Mask appearance
FG_COLOR = (255, 0, 0)        # object color (RGB)
# background color (RGB) for the middle "mask" column
BG_COLOR = (0, 0, 0)
OVERLAY_ALPHA = 0.45          # for 'fill' overlay
OVERLAY_MODE = "edge"         # 'fill' or 'edge'
EDGE_THICKNESS = 3            # px for 'edge' mode

# Polarity & QC
INVERT_MASK = False           # manual fallback
AUTO_INVERT = True            # auto-detect polarity per mask
FG_LOW = 0.02                 # if fg ratio < FG_LOW -> likely empty / wrong polarity
FG_HIGH = 0.98                # if fg ratio > FG_HIGH -> likely full / wrong polarity
# if True, skip pairs with ~empty/~full masks after auto-fix
SKIP_NEAR_EMPTY_OR_FULL = False

# Titles / labels
SHOW_ROW_LABEL_LEFT = True    # adds class label text on left margin per row
SUPTITLE = "Image | Mask | Overlay"
# ------------------------------------------------------
# Skips near-empty or near-full masks
SKIP_NEAR_EMPTY_OR_FULL = True

# Mask cleaning (applied after binarization + polarity)
CLEAN_WITH_MORPH = True
MORPH_OPEN_K = 3         # 3x3 open to remove speckles
MORPH_CLOSE_K = 3        # 3x3 close to seal tiny gaps
MIN_AREA_FRAC = 0.001    # drop components smaller than this fraction of image area
MAX_AREA_FRAC = 0.80     # drop components larger than this fraction of image area

# Collage options
LAYOUT_MODE = "overlay_only"   # 'overlay_only' or 'panel_3col'
GRID_COLS = 5                   # when overlay_only, arrange items in GRID_COLS columns
MAX_CLASSES = 10                # limit to first N classes for collage


def normalize_stem(name: str) -> str:
    stem = Path(name).stem.lower()
    stem = re.sub(r"^(mask_|label_|labels_|seg_|m_)+", "", stem)
    stem = re.sub(r"(_mask|-mask|_label|-label|_seg|-seg)+$", "", stem)
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def list_class_dirs(root: str) -> List[Path]:
    return sorted([p for p in Path(root).iterdir() if p.is_dir()])


def file_map_by_norm_stem(folder: Path, exts: set) -> Dict[str, List[Path]]:
    m: Dict[str, List[Path]] = {}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            key = normalize_stem(p.name)
            m.setdefault(key, []).append(p)
    return m


def resolve_mask_dir(masks_root: str, class_name: str) -> Optional[Path]:
    base = Path(masks_root)
    for c in [
        base / class_name,
        base / f"{class_name}_mask",
        base / f"{class_name}_masks",
        base / class_name.replace("images", "masks"),
        base / class_name.replace("_images", "_masks"),
    ]:
        if c.is_dir():
            return c
    return None


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def mask_to_bool(mask_img: Image.Image, auto_invert=True) -> Tuple[np.ndarray, float, bool]:
    """
    Convert mask to boolean FG; try auto-invert by FG ratio.
    Returns: (mask_bool, fg_ratio, inverted_flag)
    """
    arr = np.array(mask_img)
    if arr.ndim == 3:
        arr = np.array(mask_img.convert("L"))

    if np.issubdtype(arr.dtype, np.integer):
        fg = arr != 0
    else:
        # float or unusual→threshold at mid
        fg = arr >= 128

    fg_ratio = float(fg.mean())
    inverted = False

    if auto_invert:
        if fg_ratio < FG_LOW or fg_ratio > FG_HIGH:
            # try flipping
            fg2 = ~fg
            fg_ratio2 = float(fg2.mean())
            # choose the one closer to mid (less extreme)
            if abs(0.5 - fg_ratio2) < abs(0.5 - fg_ratio):
                fg, fg_ratio, inverted = fg2, fg_ratio2, True

    # manual override if requested
    if INVERT_MASK:
        fg = ~fg
        fg_ratio = float(fg.mean())
        inverted = True

    return fg, fg_ratio, inverted


def recolor_mask(mask_bool: np.ndarray, fg_color=(255, 0, 0), bg_color=(0, 0, 0)) -> Image.Image:
    h, w = mask_bool.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[~mask_bool] = bg_color
    out[mask_bool] = fg_color
    return Image.fromarray(out, mode="RGB")


def overlay_fill(image_rgb: Image.Image, mask_bool: np.ndarray, color=(255, 0, 0), alpha=0.5) -> Image.Image:
    img = np.array(image_rgb).astype(np.float32)
    color_arr = np.zeros_like(img)
    color_arr[..., 0], color_arr[..., 1], color_arr[..., 2] = color
    mask3 = mask_bool[..., None].repeat(3, axis=2)
    blended = img.copy()
    blended[mask3] = (1 - alpha) * img[mask3] + alpha * color_arr[mask3]
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def overlay_edge(image_rgb: Image.Image, mask_bool: np.ndarray, color=(255, 0, 0), thickness=3) -> Image.Image:
    """
    Draw a colored contour around the mask (clean & readable).
    """
    try:
        img = np.array(image_rgb).copy()
        m = (mask_bool.astype(np.uint8) * 255)
        # smoother chain; then small polygon simplification for nicer edges
        contours, _ = cv2.findContours(
            m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        simp = []
        for c in contours:
            peri = cv2.arcLength(c, True)
            eps = max(1e-6, 0.002 * peri)
            c2 = cv2.approxPolyDP(c, eps, True)
            if len(c2) >= 3:
                simp.append(c2)
        contours = simp if simp else contours
        cv2.drawContours(img, contours, -1, color, thickness)
        return Image.fromarray(img)
    except Exception:
        # Fallback: crude edge by morphological XOR
        from scipy.ndimage import binary_erosion
        edge = mask_bool ^ binary_erosion(mask_bool)
        img = np.array(image_rgb).copy()
        img[edge] = color
        return Image.fromarray(img)


def clean_mask_bool(mask_bool: np.ndarray) -> np.ndarray:
    """Clean mask using simple morphology and area gates. Returns boolean mask.
    - Morph open/close to remove speckles and seal tiny gaps
    - Keep components within [MIN_AREA_FRAC, MAX_AREA_FRAC] of image area
    - Fallback: if nothing remains, keep the largest component (if any)
    """
    h, w = mask_bool.shape
    if _HAS_CV2 and CLEAN_WITH_MORPH:
        m = (mask_bool.astype(np.uint8) * 255)
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(1, MORPH_OPEN_K), max(1, MORPH_OPEN_K)))
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(1, MORPH_CLOSE_K), max(1, MORPH_CLOSE_K)))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    else:
        m = (mask_bool.astype(np.uint8) * 255)

    if _HAS_CV2:
        num, labels, stats, _ = cv2.connectedComponentsWithStats(
            (m > 0).astype(np.uint8), 8)
        if num <= 1:
            return mask_bool
        area_img = float(h * w)
        keep = np.zeros_like(m)
        min_a = max(1.0, MIN_AREA_FRAC * area_img)
        max_a = MAX_AREA_FRAC * area_img
        for i in range(1, num):
            a = float(stats[i, cv2.CC_STAT_AREA])
            if a >= min_a and a <= max_a:
                keep[labels == i] = 255
        if keep.any():
            return (keep > 0)
        # Fallback: keep the largest component
        if num > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            i_best = int(np.argmax(areas) + 1)
            keep[:] = 0
            keep[labels == i_best] = 255
            return (keep > 0)
        return (m > 0)
    else:
        return (m > 0)


def pick_pairs_for_class(img_dir: Path, msk_dir: Path, n: int) -> List[Tuple[Path, Path]]:
    img_map = file_map_by_norm_stem(img_dir, IMAGE_EXTS)
    msk_map = file_map_by_norm_stem(msk_dir, MASK_EXTS)
    common_keys = sorted(set(img_map) & set(msk_map))
    if not common_keys:
        return []
    random.shuffle(common_keys)
    pairs: List[Tuple[Path, Path]] = []
    # a few extras in case we skip some later
    for key in common_keys[: max(n, 1) * 3]:
        pairs.append(
            (random.choice(img_map[key]), random.choice(msk_map[key])))
        if len(pairs) >= n * 3:
            break
    return pairs


def short_name(name: str) -> str:
    # Keep labels short and tidy
    return name.replace("_", " ")


def main():
    random.seed(RANDOM_SEED)

    img_classes = list_class_dirs(IMAGES_ROOT)
    # Limit to first N classes if requested
    if MAX_CLASSES and MAX_CLASSES > 0:
        img_classes = img_classes[:MAX_CLASSES]
    if not img_classes:
        raise SystemExit(
            f"No class folders found under IMAGES_ROOT: {IMAGES_ROOT}")

    classes: List[Tuple[Path, Path]] = []
    for cdir in img_classes:
        mdir = resolve_mask_dir(MASKS_ROOT, cdir.name)
        if mdir is not None:
            classes.append((cdir, mdir))
        else:
            print(
                f"[WARN] No mask folder for '{cdir.name}'. Tried same name, *_mask, *_masks.")

    # Gather pairs and filter with QC
    rows = []
    for img_dir, msk_dir in classes:
        wanted = SAMPLES_PER_CLASS
        for (img_path, msk_path) in pick_pairs_for_class(img_dir, msk_dir, SAMPLES_PER_CLASS):
            # Load and prep
            img = Image.open(img_path).convert("RGB")
            img = center_square(img).resize(
                (TILE_SIZE, TILE_SIZE), Image.BILINEAR)
            m = Image.open(msk_path)
            m = center_square(m).resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)

            m_bool, fg_ratio, inverted = mask_to_bool(
                m, auto_invert=AUTO_INVERT)
            m_bool = clean_mask_bool(m_bool)

            # Skip or keep extreme masks
            if SKIP_NEAR_EMPTY_OR_FULL and (fg_ratio < FG_LOW or fg_ratio > FG_HIGH):
                continue

            rows.append((img_dir.name, img, m_bool, fg_ratio, inverted))
            wanted -= 1
            if wanted <= 0:
                break

    total_rows = len(rows)
    if total_rows == 0:
        raise SystemExit("No valid image–mask pairs after QC.")

    # Decide layout
    if LAYOUT_MODE == "panel_3col":
        # Original 3-column panel
        fig_h_per_row = max(1.9, TILE_SIZE / 240)
        fig_w = 3 * fig_h_per_row * 1.25
        fig_h = total_rows * fig_h_per_row
        fig, axes = plt.subplots(total_rows, 3, figsize=(fig_w, fig_h))
        if total_rows == 1:
            import numpy as _np
            axes = _np.array([axes])

        if SUPTITLE:
            fig.suptitle(SUPTITLE, fontsize=14, y=0.995)

        for r, (cls, img, m_bool, fg_ratio, inverted) in enumerate(rows):
            colored = recolor_mask(m_bool, FG_COLOR, BG_COLOR)
            overlay = overlay_edge(img, m_bool, color=FG_COLOR, thickness=EDGE_THICKNESS) \
                if OVERLAY_MODE.lower() == "edge" else overlay_fill(img, m_bool, color=FG_COLOR, alpha=OVERLAY_ALPHA)

            axes[r, 0].imshow(img)
            axes[r, 0].set_axis_off()
            axes[r, 1].imshow(colored)
            axes[r, 1].set_axis_off()
            axes[r, 2].imshow(overlay)
            axes[r, 2].set_axis_off()

            qc_tag = ""
            if fg_ratio < FG_LOW:
                qc_tag = " (empty?)"
            elif fg_ratio > FG_HIGH:
                qc_tag = " (full?)"
            if inverted:
                qc_tag += " [auto↩]"

            axes[r, 0].set_title("image", fontsize=10)
            axes[r, 1].set_title(f"mask{qc_tag}", fontsize=10)
            axes[r, 2].set_title("overlay", fontsize=10)

            if SHOW_ROW_LABEL_LEFT:
                fig.text(0.02, 1 - ((r + 0.5) / total_rows),
                         short_name(cls), va="center", ha="left", fontsize=11)

        plt.tight_layout(
            rect=(0.06 if SHOW_ROW_LABEL_LEFT else 0.02, 0.02, 0.99, 0.98))
    else:
        # Compact overlay-only collage across classes
        n = len(rows)
        cols = max(1, GRID_COLS)
        rows_count = math.ceil(n / cols)
        fig_w = cols * (TILE_SIZE / 160)
        fig_h = rows_count * (TILE_SIZE / 160)
        fig, axes = plt.subplots(rows_count, cols, figsize=(fig_w, fig_h))
        if rows_count == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows_count == 1:
            axes = np.array([axes])

        for i, (cls, img, m_bool, fg_ratio, inverted) in enumerate(rows):
            r = i // cols
            c = i % cols
            overlay = overlay_edge(img, m_bool, color=FG_COLOR, thickness=EDGE_THICKNESS) \
                if OVERLAY_MODE.lower() == "edge" else overlay_fill(img, m_bool, color=FG_COLOR, alpha=OVERLAY_ALPHA)
            axes[r, c].imshow(overlay)
            axes[r, c].set_axis_off()
            axes[r, c].set_title(short_name(cls), fontsize=9)

        # Hide unused axes
        total_slots = rows_count * cols
        for i in range(n, total_slots):
            r = i // cols
            c = i % cols
            axes[r, c].set_visible(False)

        plt.tight_layout(pad=0.2)

    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=220, bbox_inches="tight")
    print(f"[OK] Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
