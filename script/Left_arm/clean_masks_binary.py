import os
import cv2
import numpy as np

# ============================================================
# USER PATHS  (EDIT THESE)
# ============================================================
INPUT_MASK_DIR = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_masks"
OUTPUT_MASK_DIR = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_masks_clean_safe"
LOG_FILE = "mask_cleaning_log.txt"

os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

# ============================================================
# LOGGING UTILITY
# ============================================================
def log(message):
    with open(LOG_FILE, "a") as f:
        f.write(message + "\n")
    print(message)


# ============================================================
# SAFE MASK PROCESSING FUNCTION
# ============================================================
def process_mask(mask_path, out_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        log(f"[ERROR] Cannot read: {mask_path}")
        return

    h, w = mask.shape

    # --------------------------------------------------------
    # 1) Binarize safely (works whether object is white or black)
    # --------------------------------------------------------
    _, bin_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)

    # --------------------------------------------------------
    # 2) Detect FULL WHITE (255) or FULL BLACK (0)
    # --------------------------------------------------------
    unique_vals = np.unique(bin_mask)

    if len(unique_vals) == 1 and unique_vals[0] == 255:
        log(f"[ALL WHITE] {mask_path}")
        # try to invert — sometimes object is black
        inv = cv2.bitwise_not(bin_mask)
        if np.unique(inv).tolist() != [0]:            # if inversion gives object
            bin_mask = inv
            log(f"[FIXED INVERSION] {mask_path}")
        else:
            log(f"[DISCARDED: ALL WHITE] {mask_path}")
            return

    if len(unique_vals) == 1 and unique_vals[0] == 0:
        log(f"[ALL BLACK] {mask_path}")
        log(f"[DISCARDED: SAM FAILED] {mask_path}")
        return

    # --------------------------------------------------------
    # 3) Fix grayscale masks (if any)
    # --------------------------------------------------------
    if len(unique_vals) > 2:
        log(f"[GRAYSCALE FIXED] {mask_path}")
        _, bin_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # --------------------------------------------------------
    # 4) Detect inverted masks (object = background)
    # --------------------------------------------------------
    white_pixels = np.sum(bin_mask == 255)
    black_pixels = np.sum(bin_mask == 0)

    if white_pixels < black_pixels:
        log(f"[INVERTED FIXED] {mask_path}")
        bin_mask = cv2.bitwise_not(bin_mask)

    # --------------------------------------------------------
    # 5) Remove only tiny components (SAFE)
    # --------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)

    sizes = stats[1:, cv2.CC_STAT_AREA]
    min_area = 500    # VERY SAFE — prevents accidental deletion

    cleaned = np.zeros_like(bin_mask)

    kept_count = 0
    for i, area in enumerate(sizes):
        if area >= min_area:
            cleaned[labels == i + 1] = 255
            kept_count += 1

    if kept_count == 0:
        log(f"[NO VALID OBJECT] {mask_path}")
        return

    # --------------------------------------------------------
    # 6) Save final clean mask
    # --------------------------------------------------------
    cv2.imwrite(out_path, cleaned)
    log(f"[CLEANED OK] {mask_path} -> {out_path}")


# ============================================================
# MAIN LOOP
# ============================================================
def run_cleaning():
    log("===== SAFE MASK CLEANING START =====")

    for folder in os.listdir(INPUT_MASK_DIR):
        folder_path = os.path.join(INPUT_MASK_DIR, folder)

        if not os.path.isdir(folder_path):
            continue

        out_folder = os.path.join(OUTPUT_MASK_DIR, folder)
        os.makedirs(out_folder, exist_ok=True)

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(".png"):
                continue

            mask_path = os.path.join(folder_path, fname)
            out_path  = os.path.join(out_folder, fname)

            process_mask(mask_path, out_path)

    log("===== CLEANING COMPLETE =====")


if __name__ == "__main__":
    run_cleaning()
