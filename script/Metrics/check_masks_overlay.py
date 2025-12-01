#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a panel for RANDOM images from ONE folder:

Each row: [ Original | Mask ]

Mask colors:
- Foreground (tree)  -> dark blue
- Background         -> light blue
"""

from pathlib import Path
from typing import List, Tuple

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import random

# ---------------------------------------------------------
# ROOT FOLDER PATHS
# ---------------------------------------------------------
IMAGE_ROOT = Path(r"E:/Santosh_master_thesis/Classified_Leaves")
MASK_ROOT = Path(r"E:/Santosh_master_thesis/Classified_Masks_binary")
OUT_ROOT = Path(r"E:/Santosh_master_thesis/Image_Mask_Triplets_Manual")

OUT_ROOT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT_ROOT / "random_masks_from_folder.png"

TARGET_SIZE = (600, 600)

FG_COLOR = np.array([0, 0, 139], dtype=np.uint8)      # dark blue
BG_COLOR = np.array([173, 216, 230], dtype=np.uint8)  # light blue

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

# how many random images to show (at least this many if possible)
N_RANDOM_IMAGES = 3

# ---------------------------------------------------------
# CONFIG: choose ONE folder (species)
# ---------------------------------------------------------
# Example: Abies_alba_Leaves folder
SINGLE_IMAGE_FOLDER = IMAGE_ROOT / "Abies_alba_Leaves"
# You can change this to any other species folder, e.g.:
# SINGLE_IMAGE_FOLDER = IMAGE_ROOT / "Betula_pendula_Leaves"


# ---------------------------------------------------------
# Helper: pretty species name from image folder name (for titles only)
# ---------------------------------------------------------
def clean_species_name_from_image(image_path: Path) -> str:
    """Return a readable species label from the parent folder name.

    Example: Classified_Leaves/Abies_alba_Leaves/... -> "Abies alba".
    """
    folder_name = image_path.parent.name  # e.g. "Abies_alba_Leaves"
    if folder_name.endswith("_Leaves"):
        folder_name = folder_name[:-7]
    return folder_name.replace("_", " ")


# ---------------------------------------------------------
# Colorize mask to RGB (Ensure binary mask)
# ---------------------------------------------------------
def colorize_mask_from_gray(mask_gray: np.ndarray) -> Image.Image:
    """
    Converts a binary mask into an RGB image, where:
    - Foreground (tree) is dark blue
    - Background is light blue
    """
    # Ensure the mask is binary (foreground as 1, background as 0)
    mask_gray = np.where(mask_gray > 0, 1, 0)  # Convert to binary (0 and 1)

    h, w = mask_gray.shape
    m_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    fg_mask = mask_gray > 0
    m_rgb[~fg_mask] = BG_COLOR  # Background color
    m_rgb[fg_mask] = FG_COLOR   # Foreground color (tree)
    return Image.fromarray(m_rgb, mode="RGB")


# ---------------------------------------------------------
# Collect RANDOM image-mask pairs from ONE folder
# ---------------------------------------------------------
def collect_triplets() -> List[Tuple[str, Image.Image, Image.Image]]:
    triplets: List[Tuple[str, Image.Image, Image.Image]] = []

    folder = SINGLE_IMAGE_FOLDER
    if not folder.is_dir():
        print(f"[ERROR] Folder does not exist: {folder}")
        return triplets

    # List all images in this folder
    image_files = sorted(
        [p for p in folder.iterdir()
         if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    )
    if not image_files:
        print(f"[ERROR] No images found in folder: {folder}")
        return triplets

    # Build list of all images that HAVE a corresponding mask
    valid_pairs = []
    for img_path in image_files:
        species_label = clean_species_name_from_image(img_path)
        species_folder = img_path.parent.name
        mask_folder = MASK_ROOT / f"{species_folder}_mask"

        if not mask_folder.is_dir():
            print(f"[WARN] Mask folder missing: {mask_folder}")
            continue

        mask_stem = f"mask_{img_path.stem}"
        candidates = list(mask_folder.glob(mask_stem + ".*"))
        if not candidates:
            # no matching mask for this image
            continue

        mask_path = candidates[0]
        valid_pairs.append((species_label, img_path, mask_path))

    if not valid_pairs:
        print(f"[ERROR] No images with matching masks found in {folder}")
        return triplets

    # Choose random subset
    k = min(N_RANDOM_IMAGES, len(valid_pairs))
    selected_pairs = random.sample(valid_pairs, k=k)

    for species_label, img_path, mask_path in selected_pairs:
        # Load image and mask, resize them to TARGET_SIZE
        orig_img = Image.open(img_path).convert("RGB").resize(TARGET_SIZE)
        m_gray = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
        m_np = np.array(m_gray)

        # Colorize the mask
        mask_img = colorize_mask_from_gray(m_np)

        triplets.append((species_label, orig_img, mask_img))

    return triplets


# ---------------------------------------------------------
# Build panel: each row is [Original | Mask]
# ---------------------------------------------------------
def build_panel(triplets: List[Tuple[str, Image.Image, Image.Image]]) -> None:
    n_triplets = len(triplets)
    if n_triplets == 0:
        print("[ERROR] No triplets collected, nothing to plot.")
        return

    n_rows = n_triplets
    n_cols = 2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.5),
        squeeze=False
    )

    for row, (species, orig_img, mask_img) in enumerate(triplets):
        ax_orig = axes[row, 0]
        ax_mask = axes[row, 1]

        ax_orig.imshow(orig_img)
        ax_orig.set_title(species, fontsize=10)
        ax_orig.axis("off")

        ax_mask.imshow(mask_img)
        ax_mask.set_title("Mask", fontsize=10)
        ax_mask.axis("off")

    plt.tight_layout()
    fig.savefig(PANEL_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved random panel to: {PANEL_PATH}")


# ---------------------------------------------------------
def main() -> None:
    triplets = collect_triplets()
    print(f"[INFO] Collected {len(triplets)} triplets")
    build_panel(triplets)


if __name__ == "__main__":
    main()
