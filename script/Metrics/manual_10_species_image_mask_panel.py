#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create ONE big panel with 2 columns of triplets (selected manually):

- Up to 10 species
- Left side: 5 species (each as [Original | Mask])
- Right side: 5 species (each as [Original | Mask])

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

# ---------------------------------------------------------
# ROOT FOLDER PATHS (defined directly in the script)
# ---------------------------------------------------------
IMAGE_ROOT = Path(
    r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data")
MASK_ROOT = Path(
    r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Masks")
OUT_ROOT = Path(
    r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Triplets")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

PANEL_PATH = OUT_ROOT / "all_species_2x5_manual_panel_legend1.png"

TARGET_SIZE = (600, 600)

FG_COLOR = np.array([0, 0, 139], dtype=np.uint8)         # dark blue
BG_COLOR = np.array([173, 216, 230], dtype=np.uint8)     # light blue


# ---------------------------------------------------------
# Helper: pretty species name from image folder name
# ---------------------------------------------------------
def clean_species_name_from_image(image_path: Path) -> str:
    """Return a readable species label from the parent folder name.

    Example: Data/010_Quercus_rubra/....jpg -> "Quercus rubra".
    """
    folder_name = image_path.parent.name  # e.g. "010_Quercus_rubra"
    parts = folder_name.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        raw = parts[1]
    else:
        raw = folder_name
    return raw.replace("_", " ")


# ---------------------------------------------------------
# Colorize mask to RGB (Ensure binary mask)
# ---------------------------------------------------------
def colorize_mask_from_gray(mask_gray: np.ndarray) -> Image.Image:
    mask_bin = (mask_gray > 0)
    fg_frac = mask_bin.mean()
    # If mask is mostly filled, invert it
    if fg_frac > 0.5:
        mask_bin = ~mask_bin

    h, w = mask_bin.shape
    m_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    m_rgb[~mask_bin] = BG_COLOR   # light blue background
    m_rgb[mask_bin] = FG_COLOR   # dark blue leaf
    return Image.fromarray(m_rgb, mode="RGB")

# ---------------------------------------------------------
# Collect one image-mask pair per species (User specifies file paths)
# ---------------------------------------------------------
def collect_triplets() -> List[Tuple[str, Image.Image, Image.Image]]:
    triplets = []

    # List the file paths for each species' image

    image_paths = [

        # Manually specify the image file paths for each species

        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/001_Abies_alba/obs_9171450_photo_12407581.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/002_Acer_pseudoplatanus/obs_260683389_photo_468191559.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/003_Betula_pendula/obs_224576666_photo_398055336.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/004_Fagus_sylvatica/obs_246838947_photo_440739682.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/005_Fraxinus_excelsior/obs_224721472_photo_398307969.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/006_Larix_decidua/obs_60697543_photo_97076614.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/007_Picea_abies/obs_234537332_photo_416959276.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/008_Pinus_sylvestris/obs_256995760_photo_460921589.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/009_Pseudotsuga_menziesii/obs_242783272_photo_432848456.jpg"),
        Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data/010_Quercus_rubra/obs_242254783_photo_431876325.jpg"),

    ]

    for img_path in image_paths:
        species_label = clean_species_name_from_image(img_path)

        # Derive mask folder from image folder name, e.g.
        # Data/010_Quercus_rubra -> Masks/010_Quercus_rubra_mask
        species_folder_name = img_path.parent.name
        mask_folder = MASK_ROOT / f"{species_folder_name}_mask"

        if not mask_folder.is_dir():
            print(
                f"Mask folder {mask_folder} does not exist, skipping species {species_label}.")
            continue

        # Derive mask filename from image stem, e.g.
        # obs_...jpg -> mask_obs_....*
        mask_stem = f"mask_{img_path.stem}"
        candidates = list(mask_folder.glob(mask_stem + ".*"))
        if not candidates:
            print(
                f"No mask matching {mask_stem}.* in {mask_folder}, skipping species {species_label}.")
            continue

        mask_path = candidates[0]

        # Load image and mask, resize them to TARGET_SIZE
        orig_img = Image.open(img_path).convert("RGB").resize(TARGET_SIZE)
        m_gray = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
        m_np = np.array(m_gray)

        # Colorize the mask
        mask_img = colorize_mask_from_gray(m_np)

        # Add to triplets
        triplets.append((species_label, orig_img, mask_img))

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
    n_cols = 4
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 3.0),
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
            axes[row, 0].set_title(f"{species_l}", fontsize=11)
            axes[row, 0].axis("off")
            axes[row, 1].imshow(mask_l)
            axes[row, 1].set_title("Mask", fontsize=11)
            axes[row, 1].axis("off")

        if right_idx < n_triplets:
            species_r, orig_r, mask_r = triplets[right_idx]
            axes[row, 2].imshow(orig_r)
            axes[row, 2].set_title(f"{species_r}", fontsize=11)
            axes[row, 2].axis("off")
            axes[row, 3].imshow(mask_r)
            axes[row, 3].set_title("Mask", fontsize=11)
            axes[row, 3].axis("off")

    plt.tight_layout(pad=0.1, w_pad=0.2, h_pad=0.1)
    plt.subplots_adjust(wspace=0.02, hspace=0.05)

    # Add legend at the bottom for colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=tuple(FG_COLOR / 255.0), label="Target species"),
        Patch(facecolor=tuple(BG_COLOR / 255.0), label="Background/other species"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

    fig.savefig(PANEL_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved 2×5 panel (no inverted masks) to: {PANEL_PATH}")


# ---------------------------------------------------------
def main() -> None:
    triplets = collect_triplets()
    print(f"[INFO] Collected {len(triplets)} triplets")
    build_panel(triplets)


if __name__ == "__main__":
    main()
