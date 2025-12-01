import os
import cv2
import glob
import random
import numpy as np

# ==============================
# USER PATHS  (EDIT)
# ==============================
IMG_ROOT   = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
MASK_ROOT  = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Masks"

OUT_IMG    = r"E:/Santosh_master_thesis/DATA_YOLOv11_left_arm_clean/images"
OUT_LABELS = r"E:/Santosh_master_thesis/DATA_YOLOv11_left_arm_clean/labels"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

TARGET_SIZE = 1024
MARGIN_FACTOR = 2.0
MAX_JITTER   = 0.08   # 8% jitter so object is not perfectly centered


# ==============================
# Species → Class ID mapping
# ==============================
SPECIES_TO_ID = {
    "001_Abies_alba": 0,
    "002_Acer_pseudoplatanus": 1,
    "003_Betula_pendula": 2,
    "004_Fagus_sylvatica": 3,
    "005_Fraxinus_excelsior": 4,
    "006_Larix_decidua": 5,
    "007_Picea_abies": 6,
    "008_Pinus_sylvestris": 7,
    "009_Pseudotsuga_menziesii": 8,
    "010_Quercus_rubra": 9
}


def species_from_path(path):
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p in SPECIES_TO_ID:
            return p
    return None


def largest_contour(mask_bin):
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def jitter_box(x, y, w, h, W, H):
    cx = x + w/2
    cy = y + h/2

    side = max(w, h) * MARGIN_FACTOR

    jx = (random.random()*2 - 1) * MAX_JITTER * w
    jy = (random.random()*2 - 1) * MAX_JITTER * h

    cx += jx
    cy += jy

    x0 = max(0, int(cx - side/2))
    y0 = max(0, int(cy - side/2))
    x1 = min(W, int(cx + side/2))
    y1 = min(H, int(cy + side/2))

    if x1 <= x0: x1 = x0 + 1
    if y1 <= y0: y1 = y0 + 1

    return x0, y0, x1, y1


def contour_to_yolo(cnt, crop_box, final_sz):
    x0, y0, x1, y1 = crop_box
    crop_w = x1 - x0
    crop_h = y1 - y0

    poly = []
    for pt in cnt[:, 0, :]:
        px, py = pt.astype(float)
        px -= x0
        py -= y0
        px = px * final_sz / crop_w
        py = py * final_sz / crop_h
        px /= final_sz
        py /= final_sz
        poly.extend([float(np.clip(px,0,1)), float(np.clip(py,0,1))])

    return poly


def process(img_path):
    species = species_from_path(img_path)
    if species is None:
        print("[SKIP] Species not found in path:", img_path)
        return

    class_id = SPECIES_TO_ID[species]

    img = cv2.imread(img_path)
    if img is None:
        print("[ERR] Cannot read", img_path)
        return
    H, W = img.shape[:2]

    # ==============================
    # Construct mask path
    # e.g. 001_Abies_alba → 001_Abies_alba_mask
    # img: obs_123.jpg → mask_obs_123.jpg
    # ==============================
    folder = os.path.basename(os.path.dirname(img_path))
    mask_folder = folder + "_mask"

    img_name = os.path.basename(img_path)
    stem, _ = os.path.splitext(img_name)
    mask_name = f"mask_{stem}.jpg"

    mask_path = os.path.join(MASK_ROOT, mask_folder, mask_name)

    if not os.path.exists(mask_path):
        print("[ERR] Mask missing:", mask_path)
        return

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("[ERR] Cannot read mask:", mask_path)
        return
    if mask.shape[:2] != (H, W):
        print("[ERR] Size mismatch for:", img_path)
        return

    _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    cnt = largest_contour(mask_bin)
    if cnt is None or cv2.contourArea(cnt) < 20:
        print("[WARN] No valid contour:", img_path)
        return

    x, y, w, h = cv2.boundingRect(cnt)
    crop_box = jitter_box(x, y, w, h, W, H)
    x0, y0, x1, y1 = crop_box

    crop_img = img[y0:y1, x0:x1]
    crop_img = cv2.resize(crop_img, (TARGET_SIZE, TARGET_SIZE))

    poly = contour_to_yolo(cnt, crop_box, TARGET_SIZE)
    if len(poly) < 6:
        print("[WARN] Bad polygon:", img_path)
        return

    rel = os.path.relpath(img_path, IMG_ROOT)
    out_img = os.path.join(OUT_IMG, rel)
    out_lbl = os.path.join(OUT_LABELS, os.path.splitext(rel)[0] + ".txt")

    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    os.makedirs(os.path.dirname(out_lbl), exist_ok=True)

    cv2.imwrite(out_img, crop_img)

    with open(out_lbl, "w", encoding="utf-8") as f:
        f.write(str(class_id) + " " + " ".join(f"{v:.6f}" for v in poly) + "\n")

    print("[OK]", img_path)


def main():
    imgs = glob.glob(os.path.join(IMG_ROOT, "**", "*.jpg"), recursive=True)
    imgs += glob.glob(os.path.join(IMG_ROOT, "**", "*.jpeg"), recursive=True)
    imgs += glob.glob(os.path.join(IMG_ROOT, "**", "*.jpg"), recursive=True)

    print("Found", len(imgs), "images.")
    
    for p in imgs:
        process(p)


if __name__ == "__main__":
    main()
