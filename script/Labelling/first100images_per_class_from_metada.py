import os
import csv
import shutil

# === CONFIG ===
# Path to your metadata CSV
csv_path = "E:/Santosh_master_thesis/prediction_metadata_LOT.csv"
output_root = "./100_sorted_images_with_LOT"         # Where to copy sorted images

os.makedirs(output_root, exist_ok=True)

# Counter for each class (no species breakdown)
class_counts = {}

# === Read CSV and copy images ===
with open(csv_path, newline='') as f:
    reader = csv.DictReader(f)
    for row in reader:
        img_path = row["image_path"]
        pred_class = row["predicted_class"]

        # Init per-class counter
        if pred_class not in class_counts:
            class_counts[pred_class] = 0

        # Only copy if less than 100 images for this class
        if class_counts[pred_class] < 100:
            class_dir = os.path.join(output_root, pred_class)
            os.makedirs(class_dir, exist_ok=True)
            try:
                shutil.copy(img_path, class_dir)
                class_counts[pred_class] += 1
            except Exception as e:
                print(f"Failed to copy {img_path}: {e}")

print("✅ First 100 images per class copied to respective class folders.")
