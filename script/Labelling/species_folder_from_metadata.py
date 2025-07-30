import os
import csv
import shutil

# === CONFIG ===
csv_path = "E:/Santosh_master_thesis/prediction_metadata.csv"  # Path to your metadata CSV
output_root = "./Species_folder_sorted_images"         # Where to copy sorted images

os.makedirs(output_root, exist_ok=True)

# === Read CSV and copy images ===
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row["image_path"]
        pred_class = row["predicted_class"]

        # Extract species from the parent folder of the image
        species = os.path.basename(os.path.dirname(img_path))

        # Create class/species folder if it doesn't exist
        class_species_dir = os.path.join(output_root, pred_class, species)
        os.makedirs(class_species_dir, exist_ok=True)

        # Copy image
        try:
            shutil.copy(img_path, class_species_dir)
        except Exception as e:
            print(f"Failed to copy {img_path}: {e}")

print("✅ Images sorted into folders by predicted class.")