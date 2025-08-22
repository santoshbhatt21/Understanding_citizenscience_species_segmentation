import os
import shutil

# === CONFIG ===
# Path to your skipped images log
skipped_log_path = "E:/Santosh_master_thesis/skipped_images.log"
output_root = "./Low_confidence_sorted_images"  # Where to copy sorted images
low_conf_class = "LowConfidence"

os.makedirs(output_root, exist_ok=True)

# Counter for each species
species_counts = {}
max_per_species = 50  # Change as needed

with open(skipped_log_path, "r") as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 1:
            continue
            continue
        img_path = parts[0]
        if not os.path.isfile(img_path):
            continue
        # Extract species from the parent folder of the image
        species = os.path.basename(os.path.dirname(img_path))
        key = species
        if key not in species_counts:
            species_counts[key] = 0
        # Only copy if less than max_per_species images for this species
        if species_counts[key] < max_per_species:
            class_dir = os.path.join(output_root, low_conf_class, species)
            os.makedirs(class_dir, exist_ok=True)
            try:
                shutil.copy(img_path, class_dir)
                species_counts[key] += 1
            except Exception as e:
                print(f"Failed to copy {img_path}: {e}")

print(f"✅ First {max_per_species} low-confidence images per species copied to '{output_root}/{low_conf_class}/<species>' folders.")
