import os
import numpy as np
from PIL import Image
import shutil

# === Input/Output Paths ===
input_mask_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_Mask_July10"         
input_label_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_Mask_July10"      # e.g., "data/labels"

# === OUTPUT ===
output_mask_root = "output/masks_uint8"
output_label_root = "output/labels"

# === Create base output dirs ===
os.makedirs(output_mask_root, exist_ok=True)
os.makedirs(output_label_root, exist_ok=True)

# === Process each mask ===
for subfolder in os.listdir(input_mask_root):
    subfolder_path = os.path.join(input_mask_root, subfolder)

    if not os.path.isdir(subfolder_path):
        continue  # skip files

    for class_folder in os.listdir(subfolder_path):
        class_folder_path = os.path.join(subfolder_path, class_folder)

        if not os.path.isdir(class_folder_path):
            continue

        for fname in os.listdir(class_folder_path):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".npy")):
                continue

            # === Input mask path ===
            input_mask_path = os.path.join(class_folder_path, fname)

            # === Load mask ===
            if fname.endswith(".npy"):
                mask = np.load(input_mask_path)
            else:
                mask = np.array(Image.open(input_mask_path)) / 255.0

            # === Unnormalize ===
            mask_uint8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)

            # Convert to RGB if grayscale
            if mask_uint8.ndim == 2:
                mask_uint8 = np.stack([mask_uint8] * 3, axis=-1)

            # === Output mask path ===
            out_mask_dir = os.path.join(output_mask_root, subfolder, class_folder)
            os.makedirs(out_mask_dir, exist_ok=True)
            out_mask_path = os.path.join(out_mask_dir, os.path.splitext(fname)[0] + ".png")

            Image.fromarray(mask_uint8).save(out_mask_path)

            # === Handle labels ===
            label_name = os.path.splitext(fname)[0] + ".txt"
            input_label_path = os.path.join(input_label_root, subfolder, class_folder, label_name)
            out_label_dir = os.path.join(output_label_root, subfolder, class_folder)
            os.makedirs(out_label_dir, exist_ok=True)

            if os.path.exists(input_label_path):
                shutil.copy(input_label_path, os.path.join(out_label_dir, label_name))
            else:
                print(f"[WARN] Label file not found for: {input_label_path}")

print("✅ All masks processed and saved as uint8 with labels.")
