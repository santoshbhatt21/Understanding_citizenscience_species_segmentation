import os
import shutil

# --- Paths ---
source_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labelling_Manual"        # Your current structure with 9 species
target_root = "flat_labeled_data"        # New output folder with just 3 folders

# --- Part class folders (must match your subfolder names exactly) ---
part_classes = ['Leaves', 'Trunks', 'Others']

# --- Create output folders ---
for cls in part_classes:
    os.makedirs(os.path.join(target_root, cls), exist_ok=True)

# --- Walk through each species and part folder ---
for species in os.listdir(source_root):
    species_path = os.path.join(source_root, species)
    if not os.path.isdir(species_path):
        continue

    for part in part_classes:
        part_path = os.path.join(species_path, part)
        if not os.path.isdir(part_path):
            continue

        for img_name in os.listdir(part_path):
            src_img = os.path.join(part_path, img_name)
            dst_img = os.path.join(target_root, part, f"{species}_{img_name}")
            shutil.copy2(src_img, dst_img)

print("✅ All labeled images flattened into 3 global folders for training.")
