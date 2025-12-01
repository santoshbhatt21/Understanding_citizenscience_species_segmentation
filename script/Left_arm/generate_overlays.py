import os
import cv2
import numpy as np
from PIL import Image

# ======================================================
# USER PATHS (EDIT THESE)
# ======================================================
IMAGES_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
MASKS_ROOT  = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_masks_clean"
OVERLAY_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Overlays_clean"


# ======================================================
# OVERLAY FUNCTION
# ======================================================
def generate_overlay(original, binary_mask):
    overlay = original.copy()
    mask_bool = binary_mask.astype(bool)

    overlay[mask_bool] = (
        overlay[mask_bool] * 0.5 + np.array([255, 0, 0]) * 0.5   # red tint
    ).astype(np.uint8)

    return overlay


# ======================================================
# MAIN LOOP
# ======================================================
def generate_all_overlays():
    species_list = [d for d in os.listdir(IMAGES_ROOT)
                    if os.path.isdir(os.path.join(IMAGES_ROOT, d))]

    for species in species_list:

        img_dir  = os.path.join(IMAGES_ROOT, species)
        mask_dir = os.path.join(MASKS_ROOT, species + "_mask")
        out_dir  = os.path.join(OVERLAY_ROOT, species)

        if not os.path.isdir(mask_dir):
            print(f"[WARNING] No cleaned mask dir for {species}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        img_files = [f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"\nProcessing overlays for: {species}")

        for img_name in img_files:
            img_base = os.path.splitext(img_name)[0]
            mask_name = f"mask_{img_base}.png"
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"[WARNING] Missing cleaned mask for {img_name}")
                continue

            # load image & mask
            original = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)), cv2.COLOR_BGR2RGB)
            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            # generate overlay
            overlay = generate_overlay(original, binary_mask)

            # save
            out_path = os.path.join(out_dir, img_name)
            cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            print(f"  [OK] {out_path}")


if __name__ == "__main__":
    generate_all_overlays()
    print("\nAll overlays saved to:", OVERLAY_ROOT)
