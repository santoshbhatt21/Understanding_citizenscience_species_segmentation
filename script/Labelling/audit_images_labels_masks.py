import os
from typing import List, Tuple

# Configure your dataset root (classes live directly under this)
BASE_DIR = r"E:/Santosh_master_thesis/LT_species_organ_10_species"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif",
              ".tiff", ".JPG", ".PNG", ".JPEG")
MASK_BG_SUFFIX = "_mask"
LABELS_SUFFIX = "_labels"


def list_files_recursive(root: str, exts: Tuple[str, ...]) -> List[str]:
    out = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(tuple(e.lower() for e in exts)):
                out.append(os.path.join(r, f))
    return out


def rel_to_class(path: str, class_root: str) -> str:
    rel = os.path.relpath(path, class_root)
    return rel.replace("\\", "/")


def audit_class(class_name: str) -> None:
    class_root = os.path.join(BASE_DIR, class_name)
    if not os.path.isdir(class_root):
        return

    # Expected sibling outputs
    mask_root = os.path.join(BASE_DIR, f"{class_name}{MASK_BG_SUFFIX}")
    labels_root = os.path.join(BASE_DIR, f"{class_name}{LABELS_SUFFIX}")

    # Gather all images recursively under class_root
    img_paths = list_files_recursive(class_root, IMAGE_EXTS)

    missing_mask, missing_label = [], []
    have_mask, have_label = 0, 0

    for ip in img_paths:
        rel = rel_to_class(ip, class_root)
        base = os.path.splitext(os.path.basename(ip))[0]
        subdir = os.path.dirname(rel)

        # Expected paths
        mask_dir = os.path.join(mask_root, subdir) if subdir else mask_root
        mask_path = os.path.join(mask_dir, f"mask_{base}.png")
        label_dir = os.path.join(
            labels_root, subdir) if subdir else labels_root
        label_path = os.path.join(label_dir, f"{base}.txt")

        if os.path.isfile(mask_path):
            have_mask += 1
        else:
            missing_mask.append(rel)

        if os.path.isfile(label_path):
            have_label += 1
        else:
            missing_label.append(rel)

    print(f"\nClass: {class_name}")
    print(f"  images: {len(img_paths)}")
    print(f"  masks:  {have_mask}  (missing: {len(missing_mask)})")
    print(f"  labels: {have_label} (missing: {len(missing_label)})")

    # Optional: write missing lists next to class root
    if missing_mask:
        with open(os.path.join(BASE_DIR, f"missing_masks_{class_name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing_mask))
    if missing_label:
        with open(os.path.join(BASE_DIR, f"missing_labels_{class_name}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(missing_label))


def main():
    classes = [d for d in sorted(os.listdir(BASE_DIR))
               if os.path.isdir(os.path.join(BASE_DIR, d))
               and not d.endswith(MASK_BG_SUFFIX)
               and not d.endswith(LABELS_SUFFIX)]

    for cls in classes:
        audit_class(cls)


if __name__ == "__main__":
    main()
