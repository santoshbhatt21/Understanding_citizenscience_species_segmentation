import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO  # Ensure you have the best trained YOLO model

# =========================
# CONFIG – EDIT THIS PART
# =========================

# Root folders for MASKS (grayscale masks: 0 = non-target, 1 = target, etc.)
MASK_ROOT_BASELINE    = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Masks"
MASK_ROOT_ORGAN_AWARE = r"E:/Santosh_master_thesis/Classified_Masks_binary"

# Where to save the panels
OUTPUT_DIR = r"E:/Santosh_master_thesis/Panels"

# Label(s) that correspond to your target species in the mask images
TARGET_LABELS = [1]          # e.g. 1 = target species

# Panel layout / size (you can tweak these)
MASK_TILE_SIZE = (320, 320)  # width, height of each mask tile in pixels
SEG_TILE_SIZE  = (320, 320)  # width, height of each segmentation tile in pixels

# Colors (R, G, B)
DARK_BLUE  = np.array([  0,  51, 102], dtype=np.uint8)  # target species
LIGHT_BLUE = np.array([102, 204, 255], dtype=np.uint8)  # background + other species

# Random seed for reproducibility (set to None for fully random)
RANDOM_SEED = 42

# ======================================
# HELPER FUNCTIONS – NO NEED TO EDIT
# ======================================

def list_images(root, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return files


def colorize_mask(mask_path):
    """
    Load a grayscale mask and convert it to RGB with:
        - target species (TARGET_LABELS) in DARK_BLUE
        - everything else in LIGHT_BLUE
    """
    mask = Image.open(mask_path).convert("L")  # grayscale
    mask_np = np.array(mask)

    h, w = mask_np.shape

    # Start with everything = light blue (background + other species)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    rgb[:] = LIGHT_BLUE

    # Target species
    target_mask = np.isin(mask_np, TARGET_LABELS)
    rgb[target_mask] = DARK_BLUE

    return Image.fromarray(rgb)


def make_mask_panel(mask_root, out_path, n_tiles=3, rows=1, cols=3,
                    tile_size=(320, 320)):
    mask_files = list_images(mask_root)
    if len(mask_files) == 0:
        raise RuntimeError(f"No mask images found in {mask_root}")

    if len(mask_files) < n_tiles:
        print(f"Warning: only {len(mask_files)} mask(s) in {mask_root}, "
              f"but {n_tiles} requested. Reusing some.")
        chosen = random.choices(mask_files, k=n_tiles)
    else:
        chosen = random.sample(mask_files, n_tiles)

    # Colorize + resize each chosen mask
    tiles = [colorize_mask(p).resize(tile_size, Image.BILINEAR)
             for p in chosen]

    panel_w = cols * tile_size[0]
    panel_h = rows * tile_size[1]

    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))

    for idx, tile in enumerate(tiles):
        r = idx // cols
        c = idx % cols
        x = c * tile_size[0]
        y = r * tile_size[1]
        panel.paste(tile, (x, y))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # 300 DPI is nice for thesis printing
    panel.save(out_path, dpi=(300, 300))
    print(f"Saved mask panel -> {out_path}")


def make_segmentation_panel(mask_root, out_path, model, n_tiles=3,
                            tile_size=(320, 320)):
    mask_files = list_images(mask_root)
    if len(mask_files) == 0:
        raise RuntimeError(f"No mask images found in {mask_root}")

    if len(mask_files) < n_tiles:
        print(f"Warning: only {len(mask_files)} mask(s) in {mask_root}, "
              f"but {n_tiles} requested. Reusing some.")
        chosen = random.choices(mask_files, k=n_tiles)
    else:
        chosen = random.sample(mask_files, n_tiles)

    # Predict with YOLO
    tiles = []
    for p in chosen:
        # Run YOLO prediction
        results = model(p)
        prediction_image = results[0].plot()  # Draw predictions on the image
        prediction_image = Image.fromarray(prediction_image)
        prediction_image = prediction_image.resize(tile_size, Image.BILINEAR)
        tiles.append(prediction_image)

    rows, cols = 1, n_tiles
    panel_w = cols * tile_size[0]
    panel_h = rows * tile_size[1]

    panel = Image.new("RGB", (panel_w, panel_h), (255, 255, 255))

    for idx, tile in enumerate(tiles):
        x = idx * tile_size[0]
        panel.paste(tile, (x, 0))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    panel.save(out_path, dpi=(300, 300))
    print(f"Saved segmentation panel -> {out_path}")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    # Load the YOLO models (Baseline YOLO model for Baseline pipeline, Organ-aware YOLO model for Organ-aware pipeline)
    baseline_model = YOLO("E:/Santosh_master_thesis/species_segmentation/yolo11_10species_seg_final/weights/best.pt")  # Path to the Baseline YOLO model
    organ_aware_model = YOLO("E:/Santosh_master_thesis/species_segmentation_leaves/yolo11_leaves_seg_final/weights/best.pt")  # Path to the Organ-aware YOLO model

    # Create Mask panels (1x3 each) for Baseline and Organ-aware pipelines
    make_mask_panel(
        MASK_ROOT_BASELINE,
        os.path.join(OUTPUT_DIR, "baseline_masks_panel.png"),
        n_tiles=3, rows=1, cols=3,
        tile_size=MASK_TILE_SIZE,
    )

    make_mask_panel(
        MASK_ROOT_ORGAN_AWARE,
        os.path.join(OUTPUT_DIR, "organ_aware_masks_panel.png"),
        n_tiles=3, rows=1, cols=3,
        tile_size=MASK_TILE_SIZE,
    )

    # Create Segmentation panels (1x3 each) with YOLO prediction for Baseline and Organ-aware pipelines
    make_segmentation_panel(
        MASK_ROOT_BASELINE,
        os.path.join(OUTPUT_DIR, "baseline_segmentations_panel.png"),
        baseline_model,
        n_tiles=3,
        tile_size=SEG_TILE_SIZE,
    )

    make_segmentation_panel(
        MASK_ROOT_ORGAN_AWARE,
        os.path.join(OUTPUT_DIR, "organ_aware_segmentations_panel.png"),
        organ_aware_model,
        n_tiles=3,
        tile_size=SEG_TILE_SIZE,
    )
