import os
import csv
import shutil
import argparse
from collections import Counter


def parse_args():
    p = argparse.ArgumentParser(
        description="Copy images into class folders (Leaves/Trunks/Others) from predictions CSV.")
    p.add_argument("--csv", default="E:/Santosh_master_thesis/prediction_metadata_LOT.csv",
                   help="Path to predictions CSV")
    p.add_argument("--out", default="./LOT_all_images_labeled",
                   help="Output root directory for class folders")
    p.add_argument("--limit", type=int, default=0,
                   help="Optional max images per class (0 = no limit)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    counts = Counter()
    errors = 0

    with open(args.csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = row.get("image_path")
            pred_class = row.get("predicted_class")
            if not img_path or not pred_class:
                continue

            # Enforce optional per-class limit
            if args.limit and counts[pred_class] >= args.limit:
                continue

            dst_dir = os.path.join(args.out, pred_class)
            os.makedirs(dst_dir, exist_ok=True)

            try:
                shutil.copy(img_path, dst_dir)
                counts[pred_class] += 1
            except Exception:
                errors += 1
                # continue on errors

    # Summary
    print("\nSummary (images copied per class):")
    total = 0
    for cls in sorted(counts.keys()):
        print(f"  {cls}: {counts[cls]}")
        total += counts[cls]
    print(f"Total copied: {total}")
    if errors:
        print(f"Warnings: {errors} copy errors (missing/locked files, etc.)")


if __name__ == "__main__":
    main()
