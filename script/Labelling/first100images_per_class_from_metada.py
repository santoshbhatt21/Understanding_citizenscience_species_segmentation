import os
import csv
import shutil

# === CONFIG ===
csv_path = "E:/Santosh_master_thesis/prediction_metadata.csv"  # Path to your metadata CSV
output_root = "./sorted_images"         # Where to copy sorted images

os.makedirs(output_root, exist_ok=True)

# Counter for each class/species
class_species_counts = {}

# === Read CSV and copy images ===
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row["image_path"]
        pred_class = row["predicted_class"]
        # Extract species from the parent folder of the image
        species = os.path.basename(os.path.dirname(img_path))

        # Counter key is (class, species)
        key = (pred_class, species)
        if key not in class_species_counts:
            class_species_counts[key] = 0

        # Only copy if less than 100 images for this class/species
        if class_species_counts[key] < 100:
            class_dir = os.path.join(output_root, pred_class, species)
            os.makedirs(class_dir, exist_ok=True)
            try:
                shutil.copy(img_path, class_dir)
                class_species_counts[key] += 1
            except Exception as e:
                print(f"Failed to copy {img_path}: {e}")

print("✅ First 100 images per class per species copied to respective folders.")