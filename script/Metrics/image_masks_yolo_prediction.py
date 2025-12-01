#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create ONE big panel with 2 columns of triplets (quality-filtered, random):

- Up to 10 species
- Left side: 5 species (each as [Original | Mask | YOLO])
- Right side: 5 species (each as [Original | Mask | YOLO])

Mask colors:
- Foreground (tree)  -> dark blue
- Background         -> light blue

Quality filters:
- Use only "good" masks based on area and bounding box size.
- Use only "nice" images based on brightness.
- Use only masks whose FOREGROUND overlaps YOLO well,
  and whose INVERTED version does NOT (to exclude inverted masks).
- Within each species, pick a RANDOM image that passes all filters.
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
IMAGE_ROOT = Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data")
MASK_ROOT  = Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Masks")
OUT_ROOT   = Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Triplets")
MODEL_PATH = r"E:/Santosh_master_thesis/species_segmentation/yolo11_10species_seg_final/weights/best.pt"

OUT_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MASK_EXTS  = {".png", ".jpg", ".jpeg"}

PANEL_PATH = OUT_ROOT / "all_species_2x5_triplets_no_inverted_masks_legend.png"

TARGET_SIZE = (600, 600)

FG_COLOR = np.array([0, 0, 139], dtype=np.uint8)         # dark blue
BG_COLOR = np.array([173, 216, 230], dtype=np.uint8)     # light blue

# Mask area / bbox
MIN_FG_FRAC = 0.05
MAX_FG_FRAC = 0.85
MIN_BOX_FRAC = 0.05

# Image brightness (0–255)
MIN_BRIGHTNESS = 40
MAX_BRIGHTNESS = 220

# IoU thresholds to EXCLUDE inverted/pseudo masks
MIN_MASK_PRED_IOU_FG = 0.30     # foreground must overlap YOLO at least this much
MAX_MASK_PRED_IOU_INV = 0.15    # inverted mask must NOT overlap YOLO more than this
MIN_IOU_MARGIN = 0.10           # iou_fg must be at least this much better than iou_inv


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
# Load YOLO model
# ---------------------------------------------------------
model = YOLO(MODEL_PATH)


# ---------------------------------------------------------
# Mask + image quality checks
# ---------------------------------------------------------
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


def is_good_image(img: Image.Image) -> bool:
    gray = np.array(img.convert("L"))
    mean_brightness = gray.mean()
    if mean_brightness < MIN_BRIGHTNESS or mean_brightness > MAX_BRIGHTNESS:
        return False
    return True


def colorize_mask_from_gray(mask_gray: np.ndarray) -> Image.Image:
    h, w = mask_gray.shape
    m_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    fg_mask = mask_gray > 0
    m_rgb[~fg_mask] = BG_COLOR
    m_rgb[fg_mask]  = FG_COLOR
    return Image.fromarray(m_rgb, mode="RGB")


# ---------------------------------------------------------
# IoU between SAM mask and YOLO union mask
# ---------------------------------------------------------
def iou_with_prediction(mask_gray: np.ndarray, results):
    """
    Returns (iou_fg, iou_inv):
      - iou_fg: IoU between SAM foreground (mask>0) and YOLO union mask
      - iou_inv: IoU between SAM *background* (~fg) and YOLO union mask
    """
    if results[0].masks is None:
        return 0.0, 0.0

    masks_data = results[0].masks.data  # (N,H,W)
    union_pred = masks_data.max(dim=0)[0].cpu().numpy()
    union_img = Image.fromarray((union_pred * 255).astype(np.uint8))
    union_resized = np.array(union_img.resize(mask_gray.shape[::-1]))

    pred_fg = union_resized > 128
    sam_fg = mask_gray > 0
    sam_bg = ~sam_fg

    inter_fg = np.logical_and(sam_fg, pred_fg).sum()
    union_fg = np.logical_or(sam_fg, pred_fg).sum()
    iou_fg = inter_fg / union_fg if union_fg > 0 else 0.0

    inter_inv = np.logical_and(sam_bg, pred_fg).sum()
    union_inv = np.logical_or(sam_bg, pred_fg).sum()
    iou_inv = inter_inv / union_inv if union_inv > 0 else 0.0

    return float(iou_fg), float(iou_inv)


# ---------------------------------------------------------
# Collect one RANDOM good triplet per species
# ---------------------------------------------------------
def collect_triplets(max_species: int = 10) -> List[Tuple[str, Image.Image, Image.Image, Image.Image]]:
    triplets = []

    # Optional: set seed for reproducibility
    # random.seed(42)

    for species_dir in sorted(IMAGE_ROOT.iterdir()):
        if not species_dir.is_dir():
            continue

        species_name = clean_species_name(species_dir.name)
        chosen = False

        img_candidates = [p for p in species_dir.rglob("*")
                          if p.is_file() and p.suffix.lower() in IMAGE_EXTS]

        random.shuffle(img_candidates)

        for img_path in img_candidates:
            mask_path = find_mask_for_image(img_path)
            if mask_path is None:
                continue

            orig_img = Image.open(img_path).convert("RGB").resize(TARGET_SIZE)
            if not is_good_image(orig_img):
                continue

            m_gray = Image.open(mask_path).convert("L").resize(TARGET_SIZE)
            m_np = np.array(m_gray)
            if not is_good_mask(m_np):
                continue

            results = model.predict(
                source=str(img_path),
                imgsz=1024,
                conf=0.25,
                verbose=False,
            )
            iou_fg, iou_inv = iou_with_prediction(m_np, results)

            # ---------- STRICT FILTER TO REMOVE INVERTED MASKS ----------
            if (
                iou_fg < MIN_MASK_PRED_IOU_FG or          # FG doesn't match YOLO enough
                iou_inv > MAX_MASK_PRED_IOU_INV or        # inverted mask matches YOLO too much
                (iou_fg - iou_inv) < MIN_IOU_MARGIN       # FG not clearly better than inverted
            ):
                # This sample behaves like an "inverted" or poor mask → skip
                continue
            # ------------------------------------------------------------

            print(
                f"[INFO] Using RANDOM good sample {img_path.name} for species '{species_name}' "
                f"(IoU_fg={iou_fg:.2f}, IoU_inv={iou_inv:.2f})"
            )

            mask_img = colorize_mask_from_gray(m_np)
            pred_np = results[0].plot()
            pred_img = Image.fromarray(pred_np[:, :, ::-1]).resize(TARGET_SIZE)

            triplets.append((species_name, orig_img, mask_img, pred_img))
            chosen = True
            break

        if not chosen:
            print(f"[WARN] No suitable sample found for species '{species_name}'")

        if chosen and len(triplets) >= max_species:
            break

    return triplets


# ---------------------------------------------------------
# Build 2-column (5+5) triplet panel
# ---------------------------------------------------------
def build_panel(triplets: List[Tuple[str, Image.Image, Image.Image, Image.Image]]) -> None:
    n_triplets = len(triplets)
    if n_triplets == 0:
        print("[ERROR] No triplets collected, nothing to plot.")
        return

    n_rows = int(math.ceil(n_triplets / 2.0))
    n_cols = 6
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
            species_l, orig_l, mask_l, pred_l = triplets[left_idx]
            axes[row, 0].imshow(orig_l)
            axes[row, 0].set_title(f"{species_l}", fontsize=11)
            axes[row, 0].axis("off")
            axes[row, 1].imshow(mask_l)
            axes[row, 1].set_title("Mask", fontsize=11)
            axes[row, 1].axis("off")
            axes[row, 2].imshow(pred_l)
            axes[row, 2].set_title("YOLO prediction", fontsize=11)
            axes[row, 2].axis("off")

        if right_idx < n_triplets:
            species_r, orig_r, mask_r, pred_r = triplets[right_idx]
            axes[row, 3].imshow(orig_r)
            axes[row, 3].set_title(f"{species_r}", fontsize=11)
            axes[row, 3].axis("off")
            axes[row, 4].imshow(mask_r)
            axes[row, 4].set_title("Mask", fontsize=11)
            axes[row, 4].axis("off")
            axes[row, 5].imshow(pred_r)
            axes[row, 5].set_title("YOLO prediction", fontsize=11)
            axes[row, 5].axis("off")

    # Add legend at the bottom for mask colors and YOLO panel
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=tuple(FG_COLOR / 255.0),
              label="Target species"),
        Patch(facecolor=tuple(BG_COLOR / 255.0), label="Background/other species"),
        Patch(facecolor="none", edgecolor="black",
              label="YOLO segmentation output"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )

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
