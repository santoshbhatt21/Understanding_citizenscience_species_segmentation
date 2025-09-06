import os
import argparse
from typing import List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_UNWANTED = {".npy", ".npz", ".ds_store", "thumbs.db"}


def should_remove(filename: str, remove_non_images: bool, unwanted: set) -> bool:
    ext = os.path.splitext(filename)[1].lower()
    if filename.lower() in unwanted or ext in unwanted:
        return True
    if remove_non_images and ext not in IMAGE_EXTS:
        return True
    return False


def clean_root(root: str, remove_non_images: bool, unwanted: set, dry_run: bool) -> Tuple[int, int]:
    removed = 0
    errors = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            # Only remove files, never directories
            if not os.path.isfile(full):
                continue
            if should_remove(fn, remove_non_images, unwanted):
                try:
                    if dry_run:
                        print(f"[DRY] Would remove: {full}")
                    else:
                        os.remove(full)
                        print(f"Removed: {full}")
                    removed += 1
                except Exception as e:
                    print(f"ERROR removing {full}: {e}")
                    errors += 1
    return removed, errors


def parse_args():
    p = argparse.ArgumentParser(
        description="Recursively clean unwanted files (e.g., .npy) from a dataset tree"
    )
    p.add_argument(
        "--root",
        default=r"E:/Santosh_master_thesis/DATA_YOLO11_strict_CAM_setting",  # Change as needed
        help="Root directory to clean (default: LT_species_organ_10_species)",
    )
    p.add_argument(
        "--remove-non-images",
        action="store_true",
        help="Also remove files that are not images (jpg/png/bmp/tif)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be removed without deleting",
    )
    p.add_argument(
        "--extra-ext",
        nargs="*",
        default=[],
        help="Additional extensions or filenames to remove (e.g., .txt .json .csv)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    unwanted = set([x.lower() for x in DEFAULT_UNWANTED]) | set(
        [x.lower() for x in args.extra_ext])

    if not os.path.isdir(args.root):
        raise NotADirectoryError(f"Root not found: {args.root}")

    print(f"Root: {args.root}")
    print(f"Remove non-images: {args.remove_non_images}")
    print(f"Dry run: {args.dry_run}")
    print(f"Unwanted set: {sorted(unwanted)}")

    removed, errors = clean_root(
        args.root, args.remove_non_images, unwanted, args.dry_run)
    print("-" * 60)
    print(f"Removed files: {removed}")
    print(f"Errors: {errors}")
    if args.dry_run:
        print("Nothing was deleted (dry-run). Re-run without --dry-run to apply.")


if __name__ == "__main__":
    main()
