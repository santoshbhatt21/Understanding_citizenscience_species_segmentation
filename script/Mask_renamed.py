import os
import shutil

# === Paths ===
input_label_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_mask_july14"  # Original labels with filenames like mask_*.txt
output_label_root = "Mask_labels_July14"  # New folder to save renamed .txt files

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_and_copy_labels():
    for root, dirs, files in os.walk(input_label_root):
        rel_path = os.path.relpath(root, input_label_root)
        output_dir = os.path.join(output_label_root, rel_path)
        ensure_dir(output_dir)

        for file in files:
            if file.endswith(".txt") and file.startswith("mask_"):
                new_name = file.replace("mask_", "", 1)
                src_path = os.path.join(root, file)
                dst_path = os.path.join(output_dir, new_name)
                shutil.copy2(src_path, dst_path)
                print(f"[✓] Copied: {src_path} → {dst_path}")
            elif file.endswith(".txt"):
                # Keep normal label names unchanged
                src_path = os.path.join(root, file)
                dst_path = os.path.join(output_dir, file)
                shutil.copy2(src_path, dst_path)
                print(f"[=] Copied (unchanged): {src_path} → {dst_path}")

if __name__ == "__main__":
    process_and_copy_labels()
