import os
import shutil
import random

# ========== CONFIG ========== #
SOURCE_IMAGES = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Mask_Folders_July2"      # Folder with all .jpg/.png images
SOURCE_LABELS = "./Mask_Folders_July2"      # Folder with all .txt YOLO polygon files
TARGET_ROOT = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/YOLO"      # Final output dir
SPLIT_RATIO = 0.8                 # 80% train, 20% val
SEED = 42                         # For reproducibility
# ============================ #

def create_dirs(base_dir):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

def get_image_label_pairs(image_dir, label_dir):
    image_label_pairs = []
    for root, _, files in os.walk(image_dir):
        for img_file in files:
            if img_file.endswith(('.jpg', '.png')):
                img_path = os.path.join(root, img_file)
                rel_path = os.path.relpath(img_path, image_dir)
                label_file = os.path.splitext(rel_path)[0] + ".txt"
                label_path = os.path.join(label_dir, label_file)
                if os.path.exists(label_path):
                    image_label_pairs.append((img_path, label_path, rel_path))
                else:
                    print(f"Warning: No label found for image {img_path}")
    return image_label_pairs

def split_dataset(pairs, split_ratio=0.8):
    random.seed(SEED)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)
    return pairs[:split_idx], pairs[split_idx:]

def copy_pairs(pairs, dst_dir, split):
    for img_path, lbl_path, rel_path in pairs:
        img_dst = os.path.join(dst_dir, "images", split, rel_path)
        lbl_dst = os.path.join(dst_dir, "labels", split, os.path.splitext(rel_path)[0] + ".txt")
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
        shutil.copy(img_path, img_dst)
        shutil.copy(lbl_path, lbl_dst)

def main():
    print("🛠️ Preparing YOLOv8 dataset folders...")
    create_dirs(TARGET_ROOT)

    print("🔍 Collecting image-label pairs...")
    pairs = get_image_label_pairs(SOURCE_IMAGES, SOURCE_LABELS)

    print(f"✂️ Splitting dataset ({SPLIT_RATIO*100:.0f}% train)...")
    train_pairs, val_pairs = split_dataset(pairs, SPLIT_RATIO)

    print("📁 Copying training files...")
    copy_pairs(train_pairs, TARGET_ROOT, "train")

    print("📁 Copying validation files...")
    copy_pairs(val_pairs, TARGET_ROOT, "val")

    print(f"✅ Done! YOLOv8 dataset ready at: {TARGET_ROOT}/")

if __name__ == "__main__":
    main()
