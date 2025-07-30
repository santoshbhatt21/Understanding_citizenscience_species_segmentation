import os
import shutil

# === Paths ===
input_label_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
output_label_root = "Mask_labels_July17"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Clean folder and file names
def clean_name(name):
    return name.replace("mask_", "").replace("_mask", "")

def process_and_copy_labels():
    for root, dirs, files in os.walk(input_label_root):
        rel_path = os.path.relpath(root, input_label_root)
        folder_name = os.path.basename(root)

        # ✅ Only process folders ending with '_mask'
        if not folder_name.endswith("_mask"):
            continue

        # Clean folder path
        clean_parts = [clean_name(p) for p in rel_path.split(os.sep)]
        clean_rel_path = os.path.join(*clean_parts)
        output_dir = os.path.join(output_label_root, clean_rel_path)
        ensure_dir(output_dir)

        for file in files:
            if file.endswith(".txt"):
                # Clean file name
                clean_file = clean_name(file)
                src_path = os.path.join(root, file)
                dst_path = os.path.join(output_dir, clean_file)
                shutil.copy2(src_path, dst_path)
                print(f"[✓] {src_path} → {dst_path}")

if __name__ == "__main__":
    process_and_copy_labels()
