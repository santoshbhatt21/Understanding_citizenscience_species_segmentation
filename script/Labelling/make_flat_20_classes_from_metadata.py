"""
Manual script: build 20 species-organ folders from a metadata CSV.

Edit the CONFIG section below, then run this file. It will copy images into
folders like:
  <OUT_DIR>/Abies alba leaves/
  <OUT_DIR>/Abies alba trunks/
  ... (10 species × 2 organs = 20 folders)

Expected CSV columns: image_path, predicted_class
  - predicted_class should be 'Leaves' or 'Trunks' (case-insensitive)
  - image_path's parent folder is used to infer species, e.g. '001_Abies_alba' → 'Abies alba'
"""

import os
import csv
import shutil
import re
from collections import Counter
from typing import List, Dict


# =====================
# CONFIG — EDIT THESE
# =====================
# <- set your CSV path
CSV_PATH = r"E:/Santosh_master_thesis/prediction_metadata_LOT_10_species.csv"
# <- output root folder
OUT_DIR = r"E:/Santosh_master_thesis/LT_flat_20_from_meta"
# which classes to include
INCLUDE_CLASSES = ["Leaves", "Trunks"]
# 0 = no limit; else max copies per dest class
LIMIT_PER_CLASS = 0
# True prints actions without copying
DRY_RUN = False


# =====================
# Helpers
# =====================
def read_rows(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def parent_folder(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def to_binomial(species_folder: str) -> str:
    """Convert '001_Abies_alba' or 'Abies_alba' to 'Abies alba'."""
    name = re.sub(r"^\d+_+", "", species_folder)
    parts = re.split(r"[_\s]+", name)
    if len(parts) >= 2:
        genus = parts[0].capitalize()
        species = parts[1].lower()
        return f"{genus} {species}"
    return parts[0].capitalize() if parts else species_folder


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = read_rows(CSV_PATH)

    include = {c.lower() for c in INCLUDE_CLASSES}
    counts = Counter()  # per destination class name
    copied = 0
    skipped = 0
    errors = 0

    for row in rows:
        img = (row.get("image_path") or "").strip()
        pred = (row.get("predicted_class") or "").strip()
        if not img or not pred:
            skipped += 1
            continue
        pred_l = pred.lower()
        if pred_l not in {"leaves", "trunks"}:
            # allow variations like Leaf/Trunk
            if pred_l.startswith("leave"):
                pred_l = "leaves"
            elif pred_l.startswith("trunk"):
                pred_l = "trunks"
            else:
                skipped += 1
                continue
        if pred_l not in {c.lower() for c in include}:
            skipped += 1
            continue

        species_folder = parent_folder(img)
        binomial = to_binomial(species_folder)
        dest_class = f"{binomial} {pred_l}"
        dest_dir = os.path.join(OUT_DIR, dest_class)

        # Optional per-class cap
        if LIMIT_PER_CLASS and counts[dest_class] >= LIMIT_PER_CLASS:
            continue

        try:
            if not DRY_RUN:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.copy2(img, dest_dir)
            counts[dest_class] += 1
            copied += 1
        except Exception as e:
            errors += 1
            # print a short message but keep going
            print(f"[WARN] Failed to copy: {img} -> {dest_dir} ({e})")

    # Summary
    print("\nSummary:")
    total = 0
    for cls in sorted(counts.keys()):
        n = counts[cls]
        print(f"  {cls}: {n}")
        total += n
    print(f"Total copied: {total}  | skipped: {skipped}  | errors: {errors}")
    if DRY_RUN:
        print("(DRY RUN: no files were copied)")


if __name__ == "__main__":
    main()
