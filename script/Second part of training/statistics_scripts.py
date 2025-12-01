import os
import cv2
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------
# USER PATHS
# ---------------------------------------------------------
IMG_ROOT   = r"E:\Santosh_master_thesis\Understanding_citizenscience_species_segmentation\Data"
MASK_ROOT  = r"E:\Santosh_master_thesis\Understanding_citizenscience_species_segmentation\Data_Masks_10_species"
LABEL_ROOT = r"E:\Santosh_master_thesis\DATA_YOLO11_10_species_clean_labels"

# ---------------------------------------------------------
# COUNTERS
# ---------------------------------------------------------
total_images = 0

sam_success = 0
sam_failed = 0
empty_masks = 0

label_success = 0
label_failed = 0

species_stats = defaultdict(lambda: {
    "images": 0,
    "sam_success": 0,
    "sam_failed": 0,
    "empty_masks": 0,
    "label_success": 0,
    "label_failed": 0,
})

# ---------------------------------------------------------
# HELPER
# ---------------------------------------------------------
def mask_is_empty(mask_path):
    """Check whether a PNG mask contains any foreground pixels."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return True
    return np.sum(mask) == 0


# ---------------------------------------------------------
# PROCESS DATA
# ---------------------------------------------------------
for species in os.listdir(IMG_ROOT):
    species_dir = os.path.join(IMG_ROOT, species)
    if not os.path.isdir(species_dir):
        continue

    print(f"Processing species: {species}")

    mask_dir  = os.path.join(MASK_ROOT, species + "_mask")
    label_dir = os.path.join(LABEL_ROOT, species)

    for img_name in os.listdir(species_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        total_images += 1
        species_stats[species]["images"] += 1

        base = os.path.splitext(img_name)[0]

        # ------------------------
        # Check SAM mask
        # ------------------------
        mask_path = os.path.join(mask_dir, f"mask_{base}.png")
        if os.path.exists(mask_path):
            sam_success += 1
            species_stats[species]["sam_success"] += 1

            # Empty mask check
            if mask_is_empty(mask_path):
                empty_masks += 1
                species_stats[species]["empty_masks"] += 1

        else:
            sam_failed += 1
            species_stats[species]["sam_failed"] += 1

        # ------------------------
        # Check YOLO label
        # ------------------------
        label_path = os.path.join(label_dir, f"{base}.txt")
        if os.path.exists(label_path):
            label_success += 1
            species_stats[species]["label_success"] += 1
        else:
            label_failed += 1
            species_stats[species]["label_failed"] += 1


# ---------------------------------------------------------
# PRINT SUMMARY
# ---------------------------------------------------------
print("\n========================================")
print("              GLOBAL SUMMARY")
print("========================================")
print(f"Total images          : {total_images}")
print(f"SAM success           : {sam_success}")
print(f"SAM failed            : {sam_failed}")
print(f"Empty masks           : {empty_masks}")
print(f"YOLO label success    : {label_success}")
print(f"YOLO label failed     : {label_failed}")
print("========================================")

# ---------------------------------------------------------
# PER-SPECIES SUMMARY
# ---------------------------------------------------------
print("\n========================================")
print("             PER-SPECIES SUMMARY")
print("========================================")
for sp, stats in species_stats.items():
    print(f"\nSpecies: {sp}")
    print(f"  Images          : {stats['images']}")
    print(f"  SAM success     : {stats['sam_success']}")
    print(f"  SAM failed      : {stats['sam_failed']}")
    print(f"  Empty masks     : {stats['empty_masks']}")
    print(f"  Label success   : {stats['label_success']}")
    print(f"  Label failed    : {stats['label_failed']}")
