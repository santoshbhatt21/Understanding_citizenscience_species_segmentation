#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Create ONE big panel with 2 columns of triplets (random):

- Up to 10 species
- Left side: 5 species (each as [Original | Mask])
- Right side: 5 species (each as [Original | Mask])

Mask colors:
- Foreground (tree)  -> dark blue
- Background         -> light blue
"""

from pathlib import Path
from typing import Optional, List, Tuple
import random

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
IMAGE_ROOT = Path(r"E:/Santosh_master_thesis/Classified_Leaves")
MASK_ROOT  = Path(r"E:/Santosh_master_thesis/Classified_Masks_binary")
OUT_ROOT   = Path(r"E:/Santosh_master_thesis/Duplets")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXTS  = {".png", ".jpg", ".jpeg"}

PANEL_PATH = OUT_ROOT / "all_species_2x5_duplet__masks3.png"

TARGET_SIZE = (600, 600)

FG_COLOR = np.array([0, 0, 139], dtype=np.uint8)         # dark blue
BG_COLOR = np.array([173, 216, 230], dtype=np.uint8)     # light blue

# Mask quality thresholds
MIN_FG_FRAC = 0.01   # at least 1% foreground
MAX_FG_FRAC = 0.99   # not almost all foreground
MIN_BOX_FRAC = 0.01  # bounding box covers at least 1% of image


def is_good_mask(mask_gray: np.ndarray) -> bool:
    h, w = mask_gray.shape
    fg = mask_gray > 0
    fg_frac = fg.mean()
    if fg_frac < MIN_FG_FRAC or fg_frac > MAX_FG_FRAC:
        return False

    rows = np.any(fg, axis=1)
    cols = np.any(fg, axis=0)
    if not rows.any() or not cols.any():
        return False

    r_min, r_max = np.where(rows)[0][0], np.where(rows)[0][-1]
    c_min, c_max = np.where(cols)[0][0], np.where(cols)[0][-1]
    box_h = r_max - r_min + 1
    box_w = c_max - c_min + 1
    box_area = box_h * box_w
    box_frac = box_area / float(h * w)
    if box_frac < MIN_BOX_FRAC:
        return False

    return True
# ---------------------------------------------------------
# Helper: pretty species name from folder name
# ---------------------------------------------------------
def clean_species_name(dir_name: str) -> str:
    """
    Turn folder names like '001_Abies_alba' into 'Abies alba'.
    """
    parts = dir_name.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        name = parts[1]
    else:
        name = dir_name
    return name.replace("_", " ")


# ---------------------------------------------------------
# Find mask corresponding to an image
# ---------------------------------------------------------
def find_mask_for_image(image_path: Path) -> Optional[Path]:
    stem = image_path.stem
    candidate_stems = [
        "mask_" + stem,
        stem + "_mask",
        "mask_" + stem + "_mask",
    ]
    for sub in MASK_ROOT.rglob("*"):
        if not sub.is_file():
            continue
        if sub.suffix.lower() not in MASK_EXTS:
            continue
        if sub.stem in candidate_stems:
            return sub
    return None


# ---------------------------------------------------------
# Colorize mask to RGB (Ensure binary mask)
# ---------------------------------------------------------
def colorize_mask_from_gray(mask_gray: np.ndarray) -> Image.Image:
    # Ensure the mask is binary (foreground as 1, background as 0)
    mask_gray = np.where(mask_gray > 0, 1, 0)
    
    h, w = mask_gray.shape
    m_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    fg_mask = mask_gray > 0
    m_rgb[~fg_mask] = BG_COLOR
    m_rgb[fg_mask]  = FG_COLOR
    return Image.fromarray(m_rgb, mode="RGB")


# ---------------------------------------------------------
# Collect one RANDOM image-mask pair per species
# ---------------------------------------------------------
def collect_triplets(max_species: int = 10) -> List[Tuple[str, Image.Image, Image.Image]]:
    triplets = []

    # Optional: set seed for reproducibility
    # random.seed(42)

    # Get all species directories from IMAGE_ROOT
    species_dirs = [d for d in IMAGE_ROOT.iterdir() if d.is_dir()]
    
    # Randomly shuffle and select up to 10 species
    random.shuffle(species_dirs)
    selected_species = species_dirs[:max_species]

    for species_dir in selected_species:
        species_name = clean_species_name(species_dir.name)
        chosen = False

        # Find all image files for the current species
        img_candidates = [p for p in species_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        random.shuffle(img_candidates)

        for img_path in img_candidates:
            mask_path = find_mask_for_image(img_path)
            if mask_path is None:
                continue

            orig_img = Image.open(img_path).convert("RGB").resize(TARGET_SIZE)

            # Resize mask to match image size and ensure it's binary
            m_gray = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
            m_np = np.array(m_gray)

            if not is_good_mask(m_np):
                continue  # skip tiny / empty / huge masks

            print(
            f"[INFO] Using RANDOM sample {img_path.name} for species '{species_name}'"
        )

            mask_img = colorize_mask_from_gray(m_np)
            triplets.append((species_name, orig_img, mask_img))
            chosen = True
            break

        if not chosen:
            print(f"[WARN] No suitable sample found for species '{species_name}'")

        if len(triplets) >= max_species:
            break

    return triplets


# ---------------------------------------------------------
# Build 2-column (5+5) triplet panel
# ---------------------------------------------------------
def build_panel(triplets: List[Tuple[str, Image.Image, Image.Image]]) -> None:
    n_triplets = len(triplets)
    if n_triplets == 0:
        print("[ERROR] No triplets collected, nothing to plot.")
        return

    n_rows = int(math.ceil(n_triplets / 2.0))
    n_cols = 6
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.5),
        squeeze=False
    )

    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].axis("off")

    for row in range(n_rows):
        left_idx = 2 * row
        right_idx = 2 * row + 1

        if left_idx < n_triplets:
            species_l, orig_l, mask_l = triplets[left_idx]
            axes[row, 0].imshow(orig_l)
            axes[row, 0].set_title(f"{species_l}", fontsize=9)
            axes[row, 0].axis("off")
            axes[row, 1].imshow(mask_l)
            axes[row, 1].set_title("Mask", fontsize=9)
            axes[row, 1].axis("off")

        if right_idx < n_triplets:
            species_r, orig_r, mask_r = triplets[right_idx]
            axes[row, 3].imshow(orig_r)
            axes[row, 3].set_title(f"{species_r}", fontsize=9)
            axes[row, 3].axis("off")
            axes[row, 4].imshow(mask_r)
            axes[row, 4].set_title("Mask", fontsize=9)
            axes[row, 4].axis("off")

    plt.tight_layout()
    fig.savefig(PANEL_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved 2×5 panel (no inverted masks) to: {PANEL_PATH}")


# ---------------------------------------------------------
def main() -> None:
    triplets = collect_triplets(max_species=10)
    print(f"[INFO] Collected {len(triplets)} triplets")
    build_panel(triplets)


if __name__ == "__main__":
    main()
