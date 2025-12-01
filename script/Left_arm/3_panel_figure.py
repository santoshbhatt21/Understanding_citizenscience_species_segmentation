import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# ============================================================
# USER PATHS
# ============================================================
DATA_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
MASK_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_masks_clean_safe"
SAVE_PATH = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/final_species_panel_bestSAM.png"


# ============================================================
# MASK QUALITY FILTERS
# ============================================================
def is_all_black(mask):
    return np.sum(mask) == 0

def is_all_white(mask):
    return np.sum(mask == 255) == mask.size

def touches_all_borders(mask):
    h, w = mask.shape
    return (mask[0, :].any() and mask[-1, :].any() and
            mask[:, 0].any() and mask[:, -1].any())

def mask_convexity(mask):
    pts = np.column_stack(np.where(mask > 0))
    if len(pts) < 3:
        return 0
    hull = cv2.convexHull(pts)
    hull_area = cv2.contourArea(hull)
    mask_area = np.sum(mask > 0)
    return mask_area / hull_area if hull_area > 0 else 0


def select_best_mask(raw_mask):
    """Select best connected component from raw SAM mask using area + convexity."""
    if raw_mask.ndim == 3:
        raw_mask = raw_mask[:, :, 0]

    # Convert to binary
    mask = (raw_mask > 128).astype(np.uint8)

    # Reject invalid masks
    if is_all_black(mask) or is_all_white(mask):
        return None

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return None

    max_area = np.max(stats[1:, cv2.CC_STAT_AREA])
    candidates = []

    for cid in range(1, num_labels):
        comp = (labels == cid).astype(np.uint8)
        area = stats[cid, cv2.CC_STAT_AREA]

        if area < 500:
            continue

        if touches_all_borders(comp):
            continue

        convex = mask_convexity(comp)
        norm_area = area / max_area
        score = 0.7 * norm_area + 0.3 * convex

        candidates.append((score, comp))

    if len(candidates) == 0:
        return None

    return max(candidates, key=lambda x: x[0])[1]


# ============================================================
# LOAD BEST MASK FOR ONE SPECIES
# ============================================================
def load_one_random_sample(species):
    orig_dir = os.path.join(DATA_ROOT, species)
    mask_dir = os.path.join(MASK_ROOT, species + "_mask")

    imgs = [f for f in os.listdir(orig_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(imgs) == 0:
        print("[NO IMAGES] Species:", species)
        return None

    # Try multiple random images until a valid mask is found
    for _ in range(30):
        fname = random.choice(imgs)
        stem = os.path.splitext(fname)[0]

        orig_path = os.path.join(orig_dir, fname)
        mask_path = os.path.join(mask_dir, "mask_" + stem + ".png")

        if not os.path.exists(mask_path):
            continue

        original = np.array(Image.open(orig_path).convert("RGB"))
        raw_mask = np.array(Image.open(mask_path))

        best_mask = select_best_mask(raw_mask)

        if best_mask is None:
            continue

        # Build overlay
        overlay = original.copy()
        overlay[best_mask.astype(bool)] = (
            overlay[best_mask.astype(bool)] * 0.5 +
            np.array([255, 0, 0]) * 0.5
        ).astype(np.uint8)

        return original, best_mask, overlay, species

    print(f"[NO VALID MASK FOUND] {species}")
    return None


# ============================================================
# MAKE THE 10-SPECIES PANEL (BEST SAM)
# ============================================================
def make_species_panel_bestSAM():
    species_list = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d))
    ])

    species_list = species_list[:10]

    if len(species_list) < 10:
        print("Not enough species folders.")
        return

    print("Selected species:")
    for sp in species_list:
        print(" -", sp)

    samples = []
    for sp in species_list:
        sample = load_one_random_sample(sp)
        if sample is not None:
            samples.append(sample)

    if len(samples) == 0:
        print("No valid samples found.")
        return

    samples = samples[:10]

    left_col = samples[:5]
    right_col = samples[5:10]

    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(20, 20))

    # LEFT block (5 species)
    for row, (original, mask, overlay, sp) in enumerate(left_col):
        axes[row, 0].imshow(original)
        axes[row, 0].set_title(f"{sp}\nOriginal")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask, cmap="gray")
        axes[row, 1].set_title("Best SAM Mask")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title("Overlay")
        axes[row, 2].axis("off")

    # RIGHT block (5 species)
    for row, (original, mask, overlay, sp) in enumerate(right_col):
        axes[row, 3].imshow(original)
        axes[row, 3].set_title(f"{sp}\nOriginal")
        axes[row, 3].axis("off")

        axes[row, 4].imshow(mask, cmap="gray")
        axes[row, 4].set_title("Best SAM Mask")
        axes[row, 4].axis("off")

        axes[row, 5].imshow(overlay)
        axes[row, 5].set_title("Overlay")
        axes[row, 5].axis("off")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print("Panel saved to:", SAVE_PATH)


# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    make_species_panel_bestSAM()
    print("✓ DONE — Best SAM panel created!")
