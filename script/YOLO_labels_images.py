import os
import cv2
import numpy as np
from PIL import Image, ImageFile

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Paths ===
input_mask_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"  # RGB mask images
output_label_root = "Labels/labels_yolo"  # YOLO .txt output

# === Class name to ID mapping ===
class_name_to_id = {
    "001_Conifers": 0,
    "002_Acer_pseudoplatanus": 1,
    "003_Betula_pendula": 2,
    "pic004_Fagus_sylvaticaea": 3,
    "005_Fraxinus_excelsior": 4,
    "006_Quercus_rubra": 5
}

# === Color to Class Mapping (assumed unique color per class in RGB masks) ===
# You must define your actual RGB colors if your masks use them for classes
# Example: {(R, G, B): class_id}
color_to_class = {
    (255, 0, 0): 0,  # red = conifers
    (0, 255, 0): 1,  # green = Acer
    (0, 0, 255): 2,  # blue = Betula
    (255, 255, 0): 3,  # yellow = Fagus
    (0, 255, 255): 4,  # cyan = Fraxinus
    (255, 0, 255): 5,  # magenta = Quercus
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_rgb_mask_to_yolo_polygons(mask, class_id, img_w, img_h):
    binary_mask = cv2.inRange(mask, np.array(class_id), np.array(class_id))
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        polygon = contour.squeeze().astype(float)
        if polygon.ndim == 1:
            polygon = polygon.reshape(-1, 2)
        polygon[:, 0] /= img_w  # x / width
        polygon[:, 1] /= img_h  # y / height
        polygons.append(polygon)
    return polygons

def process_image(mask_path, class_name, save_txt_path):
    try:
        mask = np.array(Image.open(mask_path).convert("RGB"))
    except Exception as e:
        print(f"[!] Failed to open image: {mask_path} | Error: {e}")
        return
    
    img_h, img_w = mask.shape[:2]
    class_id = class_name_to_id[class_name]

    with open(save_txt_path, 'w') as f:
        for color, cid in color_to_class.items():
            if cid != class_id:
                continue
            polygons = convert_rgb_mask_to_yolo_polygons(mask, color, img_w, img_h)
            for poly in polygons:
                coords = poly.flatten().tolist()
                coords_str = ' '.join([f"{c:.6f}" for c in coords])
                f.write(f"{cid} {coords_str}\n")
    print(f"[✓] Saved: {save_txt_path}")

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
                    label_name = os.path.splitext(fname)[0] + ".txt"
                    save_txt_path = os.path.join(save_folder, label_name)
                    process_image(mask_path, class_name, save_txt_path)

if __name__ == "__main__":
    traverse_folders()
