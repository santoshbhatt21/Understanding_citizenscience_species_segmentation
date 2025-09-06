#!/usr/bin/env python3
"""
Flatten a species-organized dataset into LOT class folders.

Set SRC_ROOT and DST_ROOT manually in this file.
"""

import os
import re
import shutil
import csv
from pathlib import Path

# ========= EDIT THESE PATHS =========
SRC_ROOT = Path(r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Label_Leaves_Others_Trunks_manual_500_images")
DST_ROOT = Path(r"E:/Santosh_master_thesis/flat_labeled_Leaves_Others_Trunks_1500_images")                 # destination root
MOVE_FILES = False   # set True if you want to move instead of copy
SKIP_MASKS = True    # skip mask-like files (mask_*.png, *_mask.png)
# ====================================

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOT_CLASSES = ["Leaves", "Others", "Trunks"]

def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in IMAGE_EXTS

def looks_like_mask(name: str) -> bool:
    base = name.lower()
    return base.startswith("mask_") or base.endswith("_mask" + Path(name).suffix.lower())

def normalize_class_name(folder_name: str) -> str:
    for cls in LOT_CLASSES:
        if folder_name.strip().lower() == cls.lower():
            return cls
    return ""

def make_safe_filename(species: str, src_name: str) -> str:
    species_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", species.strip())
    return f"{species_tag}___{src_name}"

def ensure_out_dirs(out_root: Path) -> None:
    for cls in LOT_CLASSES:
        (out_root / cls).mkdir(parents=True, exist_ok=True)

def iter_lot_images(species_dir: Path) -> list:
    items = []
    species_name = species_dir.name
    for sub in species_dir.iterdir():
        if not sub.is_dir():
            continue
        class_name = normalize_class_name(sub.name)
        if not class_name:
            continue
        for p in sub.rglob("*"):
            if not p.is_file():
                continue
            if not is_image_file(p):
                continue
            if SKIP_MASKS and looks_like_mask(p.name):
                continue
            items.append((class_name, species_name, p))
    return items

def copy_or_move(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if MOVE_FILES:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))

def main():
    if not SRC_ROOT.exists():
        raise SystemExit(f"[ERROR] Source root does not exist: {SRC_ROOT}")
    if SRC_ROOT == DST_ROOT:
        raise SystemExit("[ERROR] Source and destination must be different.")

    ensure_out_dirs(DST_ROOT)

    species_dirs = [p for p in sorted(SRC_ROOT.iterdir()) if p.is_dir()]
    if not species_dirs:
        raise SystemExit(f"[ERROR] No species folders found in: {SRC_ROOT}")

    rows = []
    total = 0
    for sdir in species_dirs:
        for class_name, species_name, img_path in iter_lot_images(sdir):
            new_name = make_safe_filename(species_name, img_path.name)
            dst_path = DST_ROOT / class_name / new_name
            copy_or_move(img_path, dst_path)
            rows.append({
                "class": class_name,
                "species": species_name,
                "src_path": str(img_path),
                "dst_path": str(dst_path),
            })
            total += 1

    manifest = DST_ROOT / "LOT_manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "species", "src_path", "dst_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Total images processed: {total}")
    print(f"Leaves: {sum(r['class']=='Leaves' for r in rows)}, "
          f"Others: {sum(r['class']=='Others' for r in rows)}, "
          f"Trunks: {sum(r['class']=='Trunks' for r in rows)}")
    print(f"Manifest: {manifest}")

if __name__ == "__main__":
    main()
