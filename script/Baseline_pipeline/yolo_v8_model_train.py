import os
import shutil
import random

# ========== CONFIG ========== #
SOURCE_IMAGES = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
SOURCE_LABELS = "E:/Santosh_master_thesis/DATA_YOLO11_10_species_clean_labels"
TARGET_ROOT = "E:/Santosh_master_thesis/DATA_YOLO_11_root_structure"
SPLIT_RATIO = 0.8
SEED = 42
# ============================ #

def create_dirs(base_dir):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

def get_image_label_pairs(image_dir, label_dir):
    image_label_pairs = []
    for root, _, files in os.walk(image_dir):
        for img_file in files:
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, img_file)

                # Preserve folder structure relative to `SOURCE_IMAGES`
                rel_path = os.path.relpath(img_path, image_dir)

                # Convert image extension to .txt for label file
                label_file = os.path.splitext(rel_path)[0] + ".txt"
                label_path = os.path.join(label_dir, label_file)

                if os.path.exists(label_path):
                    image_label_pairs.append((img_path, label_path, rel_path))
                else:
                    print(f"⚠️ Warning: No label for image: {rel_path}")
    return image_label_pairs

def split_dataset(pairs, split_ratio=0.8):
    random.seed(SEED)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)
    return pairs[:split_idx], pairs[split_idx:]

def copy_pairs(pairs, dst_dir, split):
    for img_path, lbl_path, rel_path in pairs:
        # Output paths for image and label
        img_dst = os.path.join(dst_dir, "images", split, rel_path)
        lbl_dst = os.path.join(dst_dir, "labels", split, os.path.splitext(rel_path)[0] + ".txt")

        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)

        shutil.copy2(img_path, img_dst)
        shutil.copy2(lbl_path, lbl_dst)

def main():
    print("🛠️ Preparing YOLOv8 dataset folders...")
    create_dirs(TARGET_ROOT)

    print("🔍 Collecting image-label pairs (handling nested folders)...")
    pairs = get_image_label_pairs(SOURCE_IMAGES, SOURCE_LABELS)
    if not pairs:
        raise RuntimeError("❌ No matching image-label pairs found. Check paths.")

    print(f"✂️ Splitting dataset: {SPLIT_RATIO*100:.0f}% train / {100 - SPLIT_RATIO*100:.0f}% val")
    train_pairs, val_pairs = split_dataset(pairs, SPLIT_RATIO)

    print(f"📁 Copying {len(train_pairs)} training pairs...")
    copy_pairs(train_pairs, TARGET_ROOT, "train")

    print(f"📁 Copying {len(val_pairs)} validation pairs...")
    copy_pairs(val_pairs, TARGET_ROOT, "val")

    print(f"✅ Done! Dataset structured at: {TARGET_ROOT}")

if __name__ == "__main__":
    main()
