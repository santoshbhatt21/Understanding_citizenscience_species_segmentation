import os
import csv
import shutil
from collections import defaultdict

# === CONFIG ===
csv_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/prediction_metadata_three_classes_auto.csv"
output_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/split_images_into_leaves_trunks_leaves_uncertain"
copy_mode = "copy"  # "copy" or "move"

os.makedirs(output_root, exist_ok=True)

class_counts = defaultdict(int)
missing = []

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    if "image_path" not in reader.fieldnames or "predicted_class" not in reader.fieldnames:
        raise ValueError(f"CSV must have columns: image_path, predicted_class. Found: {reader.fieldnames}")

    for row in reader:
        img_path = row["image_path"]
        pred_class = row["predicted_class"] or "Uncertain"  # fallback if empty

        if not img_path or not os.path.isfile(img_path):
            missing.append(img_path)
            continue

        dest_dir = os.path.join(output_root, pred_class)
        os.makedirs(dest_dir, exist_ok=True)

        try:
            if copy_mode.lower() == "move":
                shutil.move(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
            else:
                shutil.copy2(img_path, dest_dir)
            class_counts[pred_class] += 1
        except Exception as e:
            print(f"Failed to process {img_path}: {e}")

# === Report ===
overall = sum(class_counts.values())
print("\n✅ Done.")
for cls, n in sorted(class_counts.items()):
    print(f"{cls}: {n} images")
print(f"Total copied/moved: {overall}")

if missing:
    miss_file = os.path.join(output_root, "missing_images.txt")
    with open(miss_file, "w", encoding="utf-8") as m:
        m.write("\n".join(filter(None, missing)))
    print(f"⚠️ Missing files logged to: {miss_file}")