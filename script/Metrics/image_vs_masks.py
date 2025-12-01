#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-column species panel with
- filtering for "good" masks
- custom target/background colors for masks
"""

import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from skimage import measure  # pip install scikit-image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -------------------- USER SETTINGS --------------------
IMAGES_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
MASKS_ROOT  = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Masks"
OUTPUT_PATH = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/image_for_writing/2_good_masks.png"

SAMPLES_PER_CLASS = 1
TILE_SIZE = 320
RANDOM_SEED = 42

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MASK_EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

SUPTITLE = "Image / Masks"
# ------------------------------------------------------


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
            m.setdefault(normalize_stem(p.name), []).append(p)
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


def mask_is_good(mask_arr: np.ndarray,
                 min_fg_frac: float = 0.05,
                 max_fg_frac: float = 0.6,
                 min_main_component_frac: float = 0.7) -> bool:
    """
    Decide whether a mask is 'good' for inclusion.

    Assumptions:
      - mask_arr is integer labels, 0 = background, >0 = plant.
    """

    # 1) Foreground fraction
    fg = mask_arr > 0
    fg_frac = fg.mean()
    if fg_frac < min_fg_frac or fg_frac > max_fg_frac:
        return False

    # 2) Connected components on foreground
    labeled = measure.label(fg, connectivity=2)
    props = measure.regionprops(labeled)

    if not props:
        return False

    areas = np.array([p.area for p in props])
    main_area = areas.max()
    total_fg_area = fg.sum()

    if main_area / total_fg_area < min_main_component_frac:
        # too fragmented / noisy
        return False

    # 3) Bounding-box center near image center
    main_region = props[areas.argmax()]
    minr, minc, maxr, maxc = main_region.bbox
    cy = 0.5 * (minr + maxr)
    cx = 0.5 * (minc + maxc)

    h, w = mask_arr.shape[:2]
    # central 50% region
    if not (0.25 * h <= cy <= 0.75 * h and 0.25 * w <= cx <= 0.75 * w):
        return False

    return True
# --------------------------------------------------------------


def pick_pairs_for_class(img_dir: Path, msk_dir: Path, n: int) -> List[Tuple[Path, Path]]:
    img_map = file_map_by_norm_stem(img_dir, IMAGE_EXTS)
    msk_map = file_map_by_norm_stem(msk_dir, MASK_EXTS)
    common = sorted(set(img_map) & set(msk_map))
    if not common:
        return []

    random.shuffle(common)
    pairs: List[Tuple[Path, Path]] = []

    for key in common:
        # try all masks available for this key, keep only good ones
        for img_p in img_map[key]:
            for msk_p in msk_map[key]:
                # load mask, check quality
                m = Image.open(msk_p)
                m_np = np.array(m)

                # for RGB masks, convert to single channel (you can customize)
                if m_np.ndim == 3:
                    # assume single label in any channel
                    m_np = m_np[..., 0]

                if not mask_is_good(m_np):
                    continue

                pairs.append((img_p, msk_p))
                if len(pairs) >= n:
                    return pairs

    return pairs


def species_label_from_dirname(name: str) -> str:
    # strip leading "001_", "001 ", etc.
    base = name.strip()
    base = re.sub(r"^[0-9]+[_ ]+", "", base)
    base = base.replace("_", " ")

    parts = base.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {' '.join(parts[1:])}"  # Abies alba -> A. alba
    return base


def main():
    random.seed(RANDOM_SEED)

    img_classes = list_class_dirs(IMAGES_ROOT)
    if not img_classes:
        raise SystemExit(f"No class folders under: {IMAGES_ROOT}")

    classes = []
    for cdir in img_classes:
        mdir = resolve_mask_dir(MASKS_ROOT, cdir.name)
        if mdir is not None:
            classes.append((cdir, mdir))
        else:
            print(f"[WARN] No mask folder for '{cdir.name}'.")

    class_rows: List[Tuple[str, List[Tuple[Path, Path]]]] = []
    for img_dir, msk_dir in classes:
        pairs = pick_pairs_for_class(img_dir, msk_dir, SAMPLES_PER_CLASS)
        if pairs:
            class_rows.append((img_dir.name, pairs))
        else:
            print(f"[INFO] No GOOD mask found for class '{img_dir.name}' – skipped.")

    if not class_rows:
        raise SystemExit("No matching good image–mask pairs found.")

    num_classes = len(class_rows)
    mid = (num_classes + 1) // 2
    left_classes  = class_rows[:mid]
    right_classes = class_rows[mid:]

    rows_left  = sum(len(pairs) for _, pairs in left_classes)
    rows_right = sum(len(pairs) for _, pairs in right_classes)
    nrows = max(rows_left, rows_right)
    ncols = 4  # [img, mask] x 2 columns

    fig_h_per_row = max(1.8, TILE_SIZE / 240)
    fig_w_per_pair = fig_h_per_row * 1.25
    fig = plt.figure(figsize=(2 * fig_w_per_pair, nrows * fig_h_per_row))
    axes = fig.subplots(nrows, ncols)
    if nrows == 1:
        axes = np.array([axes])

    if SUPTITLE:
        fig.suptitle(SUPTITLE, fontsize=18, y=0.995)

    # ---------- 2) define new colors for target/background ---------------
   # background = light blue, target = dark blue
    BG_COLOR = (0.80, 0.90, 1.00)   # light blue
    FG_COLOR = (0.00, 0.10, 0.40)   # dark blue
    cmap = ListedColormap([BG_COLOR, FG_COLOR])
    # ---------------------------------------------------------------------

    def place_column(col_classes, col_offset):
        current_row = 0
        for cls_name, pairs in col_classes:
            label = species_label_from_dirname(cls_name)
            for img_path, msk_path in pairs:
                if current_row >= nrows:
                    break

                img = Image.open(img_path).convert("RGB")
                img = center_square(img).resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)

                m = Image.open(msk_path)
                m = center_square(m).resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
                m_np = np.array(m)

                # convert any label >0 to 1 for binary viz
                if m_np.ndim == 3:
                    m_np = m_np[..., 0]
                bin_mask = (m_np > 0).astype(np.uint8)

                ax_img = axes[current_row, col_offset]
                ax_msk = axes[current_row, col_offset + 1]

                ax_img.imshow(img)
                ax_img.set_axis_off()

                # show recolored binary mask
                ax_msk.imshow(bin_mask, cmap=cmap, vmin=0, vmax=1)
                ax_msk.set_axis_off()

                if current_row == 0:
                    ax_img.set_title("image", fontsize=12)
                    ax_msk.set_title("mask", fontsize=12)

                # label centered below this image+mask pair
                bbox_img = ax_img.get_position()
                bbox_msk = ax_msk.get_position()
                x_center = 0.5 * (bbox_img.x0 + bbox_msk.x1)
                y_label = min(bbox_img.y0, bbox_msk.y0) - 0.015
                fig.text(x_center, y_label, label, ha="center", va="top", fontsize=11)

                current_row += 1

    place_column(left_classes, 0)
    place_column(right_classes, 2)

    plt.tight_layout(rect=(0.02, 0.05, 0.98, 0.96))
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved panel with good masks to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()