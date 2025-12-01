#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a panel with RANDOM triplets from ONE folder (RELAXED FILTERS):

Each row: [Original | Mask | YOLO]

Mask colors:
- Foreground (tree)  -> dark blue
- Background         -> light blue

Relaxed version:
- No mask area / bbox filtering
- No brightness filtering
- No IoU-based filtering
- Any image with a matching mask is accepted.
"""

from pathlib import Path
from typing import Optional, List, Tuple

from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math
import random


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
IMAGE_ROOT = Path(r"E:/Santosh_master_thesis/Classified_Leaves")
MASK_ROOT  = Path(r"E:/Santosh_master_thesis/Classified_Leaves_binary")
OUT_ROOT   = Path(r"E:/Santosh_master_thesis/Organ_triplets")
MODEL_PATH = r"E:/Santosh_master_thesis/species_segmentation_leaves/yolo11_leaves_seg_final/weights/best.pt"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXTS  = {".png", ".jpg", ".jpeg"}

PANEL_PATH = OUT_ROOT / "one_folder_random_triplets_RELAXED.png"

TARGET_SIZE = (600, 600)

FG_COLOR = np.array([0, 0, 139], dtype=np.uint8)         # dark blue
BG_COLOR = np.array([173, 216, 230], dtype=np.uint8)     # light blue

# How many random images from this folder
N_RANDOM_IMAGES = 3

# Choose ONE species folder
# Example:
SINGLE_IMAGE_FOLDER = IMAGE_ROOT / "Abies_alba_Leaves"
# Change if needed, e.g.:
# SINGLE_IMAGE_FOLDER = IMAGE_ROOT / "Betula_pendula_Leaves"


# ---------------------------------------------------------
# Helper: pretty species name from folder name
# ---------------------------------------------------------
def clean_species_name(dir_name: str) -> str:
    """
    Turn folder names like '001_Abies_alba' into 'Abies alba',
    or 'Abies_alba_Leaves' into 'Abies alba'.
    """
    parts = dir_name.split("_", 1)
    if parts[0].isdigit() and len(parts) > 1:
        name = parts[1]
    else:
        name = dir_name

    # Drop trailing "_Leaves" if present
    if name.endswith("_Leaves"):
        name = name[:-7]
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
# Load YOLO model
# ---------------------------------------------------------
model = YOLO(MODEL_PATH)


# ---------------------------------------------------------
# Colorize mask from grayscale
# ---------------------------------------------------------
def colorize_mask_from_gray(mask_gray: np.ndarray) -> Image.Image:
    h, w = mask_gray.shape
    m_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    fg_mask = mask_gray > 0
    m_rgb[~fg_mask] = BG_COLOR
    m_rgb[fg_mask]  = FG_COLOR
    return Image.fromarray(m_rgb, mode="RGB")


# ---------------------------------------------------------
# Collect RANDOM triplets from ONE folder (RELAXED FILTERS)
# ---------------------------------------------------------
def collect_triplets() -> List[Tuple[str, Image.Image, Image.Image, Image.Image]]:
    triplets: List[Tuple[str, Image.Image, Image.Image, Image.Image]] = []

    folder = SINGLE_IMAGE_FOLDER
    if not folder.is_dir():
        print(f"[ERROR] Folder does not exist: {folder}")
        return triplets

    species_name = clean_species_name(folder.name)

    # All candidate images in that folder (recursive)
    img_candidates = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]
    if not img_candidates:
        print(f"[ERROR] No images found in folder: {folder}")
        return triplets

    random.shuffle(img_candidates)

    for img_path in img_candidates:
        mask_path = find_mask_for_image(img_path)
        if mask_path is None:
            # no mask found for this image
            continue

        # Load image and mask, resize them to TARGET_SIZE
        orig_img = Image.open(img_path).convert("RGB").resize(TARGET_SIZE)
        m_gray = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
        m_np = np.array(m_gray)

        mask_img = colorize_mask_from_gray(m_np)

        # YOLO prediction on original image path
        results = model.predict(
            source=str(img_path),
            imgsz=1024,
            conf=0.25,
            verbose=False,
        )

        # YOLO .plot() returns BGR numpy array
        pred_np = results[0].plot()
        pred_img = Image.fromarray(pred_np[:, :, ::-1]).resize(TARGET_SIZE)

        print(f"[INFO] Using sample {img_path.name} (species='{species_name}')")

        triplets.append((species_name, orig_img, mask_img, pred_img))

        if len(triplets) >= N_RANDOM_IMAGES:
            break

    if not triplets:
        print(f"[WARN] No samples with matching masks found in folder '{folder}'.")
    return triplets


# ---------------------------------------------------------
# Build panel: each row [Original | Mask | YOLO]
# ---------------------------------------------------------
def build_panel(triplets: List[Tuple[str, Image.Image, Image.Image, Image.Image]]) -> None:
    n_triplets = len(triplets)
    if n_triplets == 0:
        print("[ERROR] No triplets collected, nothing to plot.")
        return

    n_rows = n_triplets
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.0, n_rows * 3.5),
        squeeze=False
    )

    for row, (species, orig_img, mask_img, pred_img) in enumerate(triplets):
        # Original
        ax0 = axes[row, 0]
        ax0.imshow(orig_img)
        ax0.set_title(species, fontsize=9)
        ax0.axis("off")

        # Mask
        ax1 = axes[row, 1]
        ax1.imshow(mask_img)
        ax1.set_title("Mask", fontsize=9)
        ax1.axis("off")

        # YOLO prediction
        ax2 = axes[row, 2]
        ax2.imshow(pred_img)
        ax2.set_title("YOLO prediction", fontsize=9)
        ax2.axis("off")

    plt.tight_layout()
    fig.savefig(PANEL_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[DONE] Saved panel for one folder to: {PANEL_PATH}")


# ---------------------------------------------------------
def main() -> None:
    triplets = collect_triplets()
    print(f"[INFO] Collected {len(triplets)} triplets")
    build_panel(triplets)


if __name__ == "__main__":
    main()
