import os
import shutil
import random
from typing import List, Tuple

# ========== CONFIG (Edit paths) ========== #
# flat 10-class images
SOURCE_IMAGES = r"E:/Santosh_master_thesis/classified_Trunks"
# sibling <class>_labels folders
SOURCE_LABELS = r"E:/Santosh_master_thesis/classified_Trunks_labels_yolo"
TARGET_ROOT = r"E:/Santosh_master_thesis/YOLO11_trunks"  # output
SPLIT_RATIO = 0.8
SEED = 42
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif',
              '.tiff', '.JPG', '.PNG', '.JPEG')
# ======================================== #


def create_dirs(base_dir: str):
    for split in ["train", "val"]:
        os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)


def list_classes(root: str) -> List[str]:
    """List class folders under SOURCE_IMAGES, excluding *_labels and *_mask."""
    classes = []
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d)
        if not os.path.isdir(p):
            continue
        if d.endswith("_labels") or d.endswith("_mask"):
            continue
        classes.append(d)
    return classes


def collect_pairs_for_class(class_name: str) -> List[Tuple[str, str, str]]:
    """Return list of (img_path, lbl_path, rel_within_class) for a single class.
    Images are under <SOURCE_IMAGES>/<class_name>/... and labels under
    <SOURCE_LABELS>/<class_name>_labels/... with same relative subpath and base name.
    """
    class_img_root = os.path.join(SOURCE_IMAGES, class_name)
    class_lbl_root = os.path.join(SOURCE_LABELS, f"{class_name}_labels")
    pairs = []
    for root, _, files in os.walk(class_img_root):
        for fname in files:
            if not fname.lower().endswith(IMAGE_EXTS):
                continue
            img_path = os.path.join(root, fname)
            rel = os.path.relpath(img_path, class_img_root)
            base = os.path.splitext(fname)[0]
            rel_dir = os.path.dirname(rel)
            lbl_path = os.path.join(class_lbl_root, rel_dir, f"{base}.txt")
            if os.path.isfile(lbl_path):
                pairs.append(
                    (img_path, lbl_path, os.path.join(class_name, rel)))
            else:
                print(f"⚠️ Missing label: {class_name}/{rel}")
    return pairs


def get_image_label_pairs(image_dir: str, label_dir: str) -> List[Tuple[str, str, str]]:
    """Collect pairs across all class folders under SOURCE_IMAGES."""
    all_pairs = []
    classes = list_classes(image_dir)
    print(f"📚 Found {len(classes)} classes under images root.")
    for cls in classes:
        pairs = collect_pairs_for_class(cls)
        print(f"  {cls}: {len(pairs)} pairs")
        all_pairs.extend(pairs)
    print(f"✅ Total matched pairs: {len(all_pairs)}")
    # Save class list to target for reference
    try:
        os.makedirs(TARGET_ROOT, exist_ok=True)
        with open(os.path.join(TARGET_ROOT, "names.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(classes))
    except Exception:
        pass
    return all_pairs


def split_dataset(pairs: List[Tuple[str, str, str]], split_ratio: float = 0.8):
    random.seed(SEED)
    random.shuffle(pairs)
    split_idx = int(len(pairs) * split_ratio)
    return pairs[:split_idx], pairs[split_idx:]


def copy_pairs(pairs: List[Tuple[str, str, str]], dst_dir: str, split: str):
    for img_path, lbl_path, rel_within_class in pairs:
        # rel_within_class starts with class_name/...
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        img_dst = os.path.join(dst_dir, "images", split, rel_within_class)
        lbl_dst = os.path.join(dst_dir, "labels", split,
                               os.path.splitext(rel_within_class)[0] + ".txt")
        os.makedirs(os.path.dirname(img_dst), exist_ok=True)
        os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
        shutil.copy2(img_path, img_dst)
        shutil.copy2(lbl_path, lbl_dst)


def main():
    print("🛠️ Preparing YOLO dataset folders...")
    create_dirs(TARGET_ROOT)

    print("🔍 Collecting image-label pairs across class folders...")
    pairs = get_image_label_pairs(SOURCE_IMAGES, SOURCE_LABELS)
    if not pairs:
        raise RuntimeError(
            "❌ No matching image-label pairs found. Check paths and that <class>_labels exist.")

    print(
        f"✂️ Splitting dataset: {SPLIT_RATIO*100:.0f}% train / {100 - SPLIT_RATIO*100:.0f}% val")
    train_pairs, val_pairs = split_dataset(pairs, SPLIT_RATIO)

    print(f"📁 Copying {len(train_pairs)} training pairs...")
    copy_pairs(train_pairs, TARGET_ROOT, "train")

    print(f"📁 Copying {len(val_pairs)} validation pairs...")
    copy_pairs(val_pairs, TARGET_ROOT, "val")

    print(f"✅ Done! Dataset structured at: {TARGET_ROOT}")


if __name__ == "__main__":
    main()
