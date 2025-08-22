import os
import csv
import shutil

# === CONFIG ===
csv_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/prediction_metadata_three_classes_auto.csv"
output_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/split_images_into-class_folders"

os.makedirs(output_root, exist_ok=True)

with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row["image_path"]
        pred_class = row["predicted_class"]

        # Extract species from image path (assumes .../<species>/<image>)
        species = os.path.basename(os.path.dirname(img_path))

        # Destination folder: <output_root>/<class>/<species>/
        dest_dir = os.path.join(output_root, pred_class, species)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy image
        try:
            shutil.copy2(img_path, dest_dir)
        except Exception as e:
            print(f"Failed to copy {img_path}: {e}")

print("✅ Images sorted into class/species folders.")