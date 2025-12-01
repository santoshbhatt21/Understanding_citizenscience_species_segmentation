import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================================
# USER PATHS
# ================================
IMG_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
MASK_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_masks_clean_safe"
OUT_OVERLAY = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Mask_Quality_Overlays"
CSV_REPORT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/mask_quality_report_clean_safe.csv"

os.makedirs(OUT_OVERLAY, exist_ok=True)

# ================================
# Helper
# ================================

def overlay_mask(img, mask):
    overlay = img.copy()
    color = np.zeros_like(img)
    color[:, :, 1] = 255  # GREEN mask

    mask3 = cv2.merge([mask, mask, mask])
    overlay = np.where(mask3, cv2.addWeighted(img, 0.6, color, 0.4, 0), img)
    return overlay

# ================================
# Main
# ================================
results = []

print("\nScanning masks...\n")

for root, dirs, files in os.walk(MASK_ROOT):
    for fname in tqdm(files):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        mask_path = os.path.join(root, fname)

        # Get image path
        species_folder = os.path.basename(root).replace("_mask", "")
        img_path = os.path.join(IMG_ROOT, species_folder, fname.replace("mask_", ""))
        
        if not os.path.exists(img_path):
            results.append([fname, "IMAGE_MISSING", "-", "-", "-", "-", "-", mask_path])
            continue

        # Read image + mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if img is None or mask is None:
            results.append([fname, "CORRUPTED", "-", "-", "-", "-", "-", mask_path])
            continue

        h, w = img.shape[:2]

        # ===== Check: Shape mismatch =====
        if mask.shape[0] != h or mask.shape[1] != w:
            results.append([fname, "SHAPE_MISMATCH", "-", "-", "-", "-", "-", mask_path])
            continue

        # ===== Check: Non-binary mask =====
        unique_vals = np.unique(mask)
        if not set(unique_vals).issubset({0, 255}):
            results.append([fname, "NON_BINARY", str(unique_vals), "-", "-", "-", "-", mask_path])
            continue

        # Convert 255→1 binary
        m = (mask > 0).astype(np.uint8)

        # ===== Check: Empty mask =====
        area = m.sum()
        total = h * w
        area_ratio = area / total

        if area_ratio == 0:
            results.append([fname, "EMPTY_MASK", area, area_ratio, "-", "-", "-", mask_path])
            continue

        # ===== Check: Too small =====
        if area_ratio < 0.01:
            status = "VERY_SMALL"
        # ===== Check: Too large =====
        elif area_ratio > 0.90:
            status = "VERY_LARGE"
        else:
            status = "OK"

        # ===== Save overlay =====
        overlay = overlay_mask(img, m)
        out_path = os.path.join(OUT_OVERLAY, fname)
        cv2.imwrite(out_path, overlay)

        # Add to table
        results.append([
            fname,
            status,
            area,
            round(area_ratio, 4),
            unique_vals,
            h, w,
            mask_path
        ])

# ================================
# Save CSV
# ================================
df = pd.DataFrame(results, columns=[
    "filename", "status", "mask_area", "mask_ratio",
    "unique_values", "height", "width", "mask_path"
])

df.to_csv(CSV_REPORT, index=False)

print("\n=======================================")
print("MASK QUALITY CHECK COMPLETE")
print(f"Report saved at: {CSV_REPORT}")
print(f"Overlays saved at: {OUT_OVERLAY}")
print("=======================================\n")
