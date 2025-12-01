import os
import cv2
import numpy as np

# ======================================================
# USER PATHS (EDIT THESE)
# ======================================================
IMAGES_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
MASKS_ROOT  = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_masks_clean"
LABELS_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_labels_clean"


# ======================================================
# CONVERT MASK → YOLO POLYGON
# ======================================================
def mask_to_yolo_polygon(binary_mask, class_id=0):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None

    cnt = max(contours, key=cv2.contourArea)

    h, w = binary_mask.shape
    polygon = []

    for p in cnt:
        x, y = p[0]
        polygon.append(x / w)
        polygon.append(y / h)

    return [class_id] + polygon


def save_yolo_polygon(label, txt_path):
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    with open(txt_path, "w") as f:
        f.write(" ".join([str(x) for x in label]))


# ======================================================
# MAIN LOOP
# ======================================================
def generate_yolo_labels():
    species_list = [d for d in os.listdir(IMAGES_ROOT)
                    if os.path.isdir(os.path.join(IMAGES_ROOT, d))]

    for species in species_list:

        img_dir  = os.path.join(IMAGES_ROOT, species)
        mask_dir = os.path.join(MASKS_ROOT, species + "_mask")
        out_dir  = os.path.join(LABELS_ROOT, species)

        if not os.path.isdir(mask_dir):
            print(f"[WARNING] No cleaned mask folder for {species}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        img_files = [f for f in os.listdir(img_dir)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"\nGenerating YOLO labels for: {species}")

        for img_name in img_files:
            base = os.path.splitext(img_name)[0]

            mask_name = f"mask_{base}.png"
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                print(f"[WARNING] Mask missing: {mask_path}")
                continue

            # Load cleaned binary mask
            binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            yolo = mask_to_yolo_polygon(binary_mask, class_id=0)
            if yolo is None:
                print(f"[FAILED] No contour in {img_name}")
                continue

            txt_path = os.path.join(out_dir, base + ".txt")
            save_yolo_polygon(yolo, txt_path)

            print(f"  [OK] {txt_path}")


if __name__ == "__main__":
    generate_yolo_labels()
    print("\nYOLO labels saved to:", LABELS_ROOT)
