import os
import cv2
import numpy as np
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Paths ===
input_mask_root = "E:/Santosh_master_thesis/LOT_masks_labels"  # RGB mask images
output_label_root = "LOT_labels/labels_yolo"  # YOLO .txt output
STRIP_PREFIX = "mask_"  # remove this prefix from filenames when saving labels
MIN_POLY_POINTS = 3      # require at least 3 points per polygon

# === Class name to ID mapping ===
class_name_to_id = {
    "Leaves_mask": 0,
    "Others_mask": 1,
    "Trunks_mask": 2,
}

# === Color to Class Mapping (only if your masks are RGB color-coded) ===
# For grayscale class-ID masks (0/1/2 + 255 background), leave this empty.
# If using color-coded masks, fill with real values like {(R,G,B): class_id}.
color_to_class = {}


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_binary_mask_to_yolo_polygons(binary_mask, img_w, img_h):
    # binary_mask: np.uint8 with values {0, 255}
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < MIN_POLY_POINTS:
            continue
        poly = contour.squeeze().astype(float)
        if poly.ndim == 1:
            poly = poly.reshape(-1, 2)
        poly[:, 0] /= img_w
        poly[:, 1] /= img_h
        polygons.append(poly)
    return polygons


def process_image(mask_path, class_name, save_txt_path):
    try:
        pil = Image.open(mask_path)
        arr = np.array(pil)
    except Exception as e:
        print(f"[!] Failed to open image: {mask_path} | Error: {e}")
        return

    class_id = class_name_to_id[class_name]
    polygons = []

    # Grayscale class-ID masks (SAM pipeline): arr.ndim == 2
    if arr.ndim == 2:
        img_h, img_w = arr.shape
        binary = (arr == class_id).astype(np.uint8) * 255
        polygons = convert_binary_mask_to_yolo_polygons(binary, img_w, img_h)
    else:
        # RGB color-coded masks
        img_h, img_w = arr.shape[:2]
        if not color_to_class:
            # No mapping provided; cannot extract polygons from RGB
            return
        for color, cid in color_to_class.items():
            if cid != class_id:
                continue
            bin_rgb = cv2.inRange(arr, np.array(color), np.array(color))
            polys = convert_binary_mask_to_yolo_polygons(bin_rgb, img_w, img_h)
            polygons.extend(polys)

    if not polygons:
        # Avoid creating 0 KB label files
        return

    os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
    with open(save_txt_path, 'w', encoding='utf-8') as f:
        for poly in polygons:
            if poly.shape[0] < MIN_POLY_POINTS:
                continue
            coords = poly.flatten().tolist()
            coords_str = ' '.join(f"{c:.6f}" for c in coords)
            f.write(f"{class_id} {coords_str}\n")


def traverse_folders():
    for class_name in sorted(os.listdir(input_mask_root)):
        class_path = os.path.join(input_mask_root, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name not in class_name_to_id:
            print(f"[!] Skipping unknown class folder: {class_name}")
            continue

        for root, dirs, files in os.walk(class_path):
            rel_root = os.path.relpath(root, input_mask_root)
            save_folder = os.path.join(output_label_root, rel_root)
            ensure_dir(save_folder)

            for fname in files:
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    mask_path = os.path.join(root, fname)
                    base = os.path.splitext(fname)[0]
                    # drop prefix (case-insensitive)
                    if base[:len(STRIP_PREFIX)].lower() == STRIP_PREFIX:
                        base = base[len(STRIP_PREFIX):]
                    label_name = base + ".txt"
                    save_txt_path = os.path.join(save_folder, label_name)
                    process_image(mask_path, class_name, save_txt_path)


if __name__ == "__main__":
    traverse_folders()
