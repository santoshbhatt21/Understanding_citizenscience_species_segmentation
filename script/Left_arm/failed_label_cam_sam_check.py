import os
from pathlib import Path

# ============================================================
# USER PATHS
# ============================================================
IMG_ROOT   = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
LABEL_ROOT = r"E:/Santosh_master_thesis/Santosh_master_thesis/DATA_YOLO11_10_species_clean_labels"
SAM_ROOT   = r"E:/Santosh_master_thesis/Santosh_master_thesis/DATA_YOLO11_10_species_clean_labels/_overlays"

# ============================================================
# Helper to extract basename WITHOUT species prefix
# ============================================================
def clean_name(name):
    """
    Removes species prefix like: 001_Abies_alba_
    Example:
        '001_Abies_alba_obs_4047575_photo_4792259.jpg'
        -> 'obs_4047575_photo_4792259'
    """
    parts = name.split("_obs_")
    if len(parts) == 2:
        return "obs_" + parts[1].split('.')[0]
    return Path(name).stem


def list_clean_basenames(root, exts):
    clean = set()
    for r, _, f in os.walk(root):
        for fname in f:
            if fname.lower().endswith(exts):
                clean.add(clean_name(fname))
    return clean


# ============================================================
# Collect names
# ============================================================
image_names = list_clean_basenames(IMG_ROOT, (".jpg", ".jpeg", ".png"))
label_names = list_clean_basenames(LABEL_ROOT, (".txt",))
sam_names   = list_clean_basenames(SAM_ROOT, (".jpg", ".png"))

# ============================================================
# Counters
# ============================================================
total_images   = len(image_names)
sam_success    = 0
sam_failed     = 0
gradcam_failed = 0
polygon_failed = 0
saved_labels   = len(label_names)

# ============================================================
# Analysis
# ============================================================
for img in image_names:

    has_label = img in label_names
    has_sam   = img in sam_names

    if has_label and has_sam:
        sam_success += 1
        continue

    if not has_sam:
        sam_failed += 1

    if not has_sam and not has_label:
        gradcam_failed += 1

    if has_sam and not has_label:
        polygon_failed += 1

# ============================================================
# Output
# ============================================================
print("\n========================================")
print("                SUMMARY")
print("========================================")
print(f"Total images          : {total_images}")
print(f"SAM success           : {sam_success}")
print(f"SAM failed (no mask)  : {sam_failed}")
print(f"GradCAM failed        : {gradcam_failed}")
print(f"Polygon failed        : {polygon_failed}")
print(f"Saved YOLO labels     : {saved_labels}")
print("========================================\n")
