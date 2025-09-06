import os
from collections import defaultdict


DATA_ROOT = r"E:\Santosh_master_thesis\DATA_YOLO11_LT"


def check_pairs() -> int:
    errors = 0
    for split in ("train", "val"):
        img_root = os.path.join(DATA_ROOT, "images", split)
        lbl_root = os.path.join(DATA_ROOT, "labels", split)
        if not os.path.isdir(img_root) or not os.path.isdir(lbl_root):
            print(f"Skip {split}: missing dir")
            continue
        subdirs = sorted(d for d in os.listdir(img_root)
                         if os.path.isdir(os.path.join(img_root, d)))
        for d in subdirs:
            img_dir = os.path.join(img_root, d)
            lbl_dir = os.path.join(lbl_root, d)
            if not os.path.isdir(lbl_dir):
                print(f"[MISSING DIR] {split}/{d}: labels dir not found")
                errors += 1
                continue
            imgs = {os.path.splitext(f)[0] for f in os.listdir(img_dir) if os.path.splitext(f)[
                1].lower() in {".jpg", ".jpeg", ".png", ".bmp"}}
            lbls = {os.path.splitext(f)[0] for f in os.listdir(
                lbl_dir) if f.lower().endswith('.txt')}
            missing_lbl = imgs - lbls
            orphan_lbl = lbls - imgs
            if missing_lbl:
                print(
                    f"[MISSING LABELS] {split}/{d}: {len(missing_lbl)} files, e.g., {sorted(list(missing_lbl))[:5]}")
                errors += len(missing_lbl)
            if orphan_lbl:
                print(
                    f"[ORPHAN LABELS] {split}/{d}: {len(orphan_lbl)} files, e.g., {sorted(list(orphan_lbl))[:5]}")
                errors += len(orphan_lbl)
    return errors


def main():
    errs = check_pairs()
    if errs == 0:
        print("OK: All images have matching labels in flattened layout.")
    else:
        print(f"DONE with issues: {errs} problems found.")


if __name__ == "__main__":
    main()
