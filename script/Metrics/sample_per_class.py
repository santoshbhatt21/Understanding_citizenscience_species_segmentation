#!/usr/bin/env python3
"""
Make a two-column collage per class: [image | mask].

- Left column: a random image for each class
- Right column: the corresponding mask (matched by filename stem, robust to common
  prefixes/suffixes like 'mask_', '_mask', 'label_', etc.)
- Saves a single figure to OUTPUT_PATH.

Usage (edit paths below or run with your own):
    python make_image_mask_panel.py
"""

import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# -------------------- USER SETTINGS --------------------
IMAGES_ROOT = r"E:/Santosh_master_thesis/classified_Leaves"     # classes as subfolders
MASKS_ROOT  = r"E:/Santosh_master_thesis/Mask_Leaves"      # same class subfolders
OUTPUT_PATH = r"E:/Santosh_master_thesis/image_for_writing/image_mask_overlay.png"

SAMPLES_PER_CLASS = 1         # set >1 to show multiple rows per class
TILE_SIZE = 320               # each tile is center-cropped square then resized
RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
MASK_EXTS  = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ------------------------------------------------------


def normalize_stem(name: str) -> str:
    """Normalize a filename stem to match images<->masks even if they have
    prefixes/suffixes like mask_, _mask, label_, _label, etc."""
    stem = Path(name).stem.lower()
    # remove common prefixes
    stem = re.sub(r"^(mask_|label_|labels_|seg_|m_)+", "", stem)
    # remove common suffixes
    stem = re.sub(r"(_mask|-mask|_label|-label|_seg|-seg)+$", "", stem)
    # collapse non-alnum to single underscore
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


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    return img.crop((left, top, left + s, top + s))


def pick_pairs_for_class(img_dir: Path, msk_dir: Path, n: int) -> List[Tuple[Path, Path]]:
    img_map = file_map_by_norm_stem(img_dir, IMAGE_EXTS)
    msk_map = file_map_by_norm_stem(msk_dir, MASK_EXTS)
    common_keys = sorted(set(img_map) & set(msk_map))
    if not common_keys:
        return []

    pairs = []
    random.shuffle(common_keys)
    for key in common_keys:
        # pick a random file if multiple with same stem
        img_path = random.choice(img_map[key])
        msk_path = random.choice(msk_map[key])
        pairs.append((img_path, msk_path))
        if len(pairs) >= n:
            break
    return pairs


def main():
    random.seed(RANDOM_SEED)

    img_classes = list_class_dirs(IMAGES_ROOT)
    if not img_classes:
        raise SystemExit(f"No class folders found under IMAGES_ROOT: {IMAGES_ROOT}")

    # keep only classes that also exist under MASKS_ROOT
    classes = []
    for cdir in img_classes:
        mdir = Path(MASKS_ROOT) / cdir.name
        if mdir.is_dir():
            classes.append((cdir, mdir))
        else:
            print(f"[WARN] No matching mask folder for class '{cdir.name}', skipping.")

    rows = sum(
        len(pick_pairs_for_class(cdir, mdir, SAMPLES_PER_CLASS))
        for cdir, mdir in classes
    )
    if rows == 0:
        raise SystemExit("No matching image–mask pairs found across classes.")

    # Prepare figure: 2 columns (image | mask)
    fig_h_per_row = max(1.8, TILE_SIZE / 240)  # heuristic sizing
    fig, axes = plt.subplots(rows, 2, figsize=(2 * fig_h_per_row * 1.2, rows * fig_h_per_row))
    if rows == 1:
        axes = np.array([axes])  # ensure 2D array

    r = 0
    for img_dir, msk_dir in classes:
        pairs = pick_pairs_for_class(img_dir, msk_dir, SAMPLES_PER_CLASS)
        for (img_path, msk_path) in pairs:
            # Left: image
            img = Image.open(img_path).convert("RGB")
            img = center_square(img).resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
            axes[r, 0].imshow(img)
            axes[r, 0].set_axis_off()
            axes[r, 0].set_title(f"{img_dir.name}  (image)")

            # Right: mask (kept as-is; if RGB stored masks, this still works)
            m = Image.open(msk_path)
            # convert to 'L' if it's single-channel; if already RGB it will display fine
            if m.mode not in ("L", "RGB"):
                try:
                    m = m.convert("L")
                except Exception:
                    m = m.convert("RGB")
            m = center_square(m).resize((TILE_SIZE, TILE_SIZE), Image.NEAREST)
            axes[r, 1].imshow(m)
            axes[r, 1].set_axis_off()
            axes[r, 1].set_title(f"{img_dir.name}  (mask)")
            r += 1

    plt.tight_layout()
    Path(OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"[OK] Saved panel to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
