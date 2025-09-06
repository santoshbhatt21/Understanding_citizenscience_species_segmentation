import os
import shutil
import re
from typing import Optional


DATA_ROOT = r"E:\Santosh_master_thesis\DATA_YOLO11_LT"


def extract_genus(species_folder: str) -> str:
    """
    Extract genus from a species folder name like '001_Abies_alba' -> 'Abies alba'.
    Heuristics: remove leading digits and underscores; split by '_' and take first alpha token.
    Lowercase result.
    """
    name = species_folder
    # strip leading numeric prefix and underscores
    name = re.sub(r"^\d+_+", "", name)
    parts = re.split(r"[_\s]+", name)
    for p in parts:
        if p and any(c.isalpha() for c in p):
            return p.lower()
    return species_folder.lower()


def move_under_flat(root_kind: str, dry_run: bool = False) -> None:
    """
    root_kind: 'images' or 'labels'
    Walk DATA_ROOT/<root_kind>/<split>/{Leaves,Trunks}/{species}/files
    Move to DATA_ROOT/<root_kind>/<split>/{genus} {organ}/files
    organ is 'leaves' or 'trunks' (singular as requested).
    """
    assert root_kind in ("images", "labels")
    base = os.path.join(DATA_ROOT, root_kind)
    for split in ("train", "val"):
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            continue
        for organ_name in os.listdir(split_dir):
            organ_path = os.path.join(split_dir, organ_name)
            if not os.path.isdir(organ_path):
                continue
            organ_lower = organ_name.lower()
            if organ_lower not in ("leaves", "trunks", "trunk"):
                # already flattened folders or other dirs; skip
                continue
            # normalize target organ name
            target_organ = "leaves" if organ_lower.startswith(
                "leave") else "trunk"

            for species_folder in os.listdir(organ_path):
                species_path = os.path.join(organ_path, species_folder)
                if not os.path.isdir(species_path):
                    continue
                genus = extract_genus(species_folder)
                dst_dir = os.path.join(split_dir, f"{genus} {target_organ}")
                os.makedirs(dst_dir, exist_ok=True)

                for fname in os.listdir(species_path):
                    src = os.path.join(species_path, fname)
                    if not os.path.isfile(src):
                        continue
                    dst = os.path.join(dst_dir, fname)
                    if dry_run:
                        print(f"Would move: {src} -> {dst}")
                    else:
                        # If destination exists, overwrite to keep latest
                        shutil.move(src, dst)

    # Optionally prune empty Leaves/Trunks trees
    for split in ("train", "val"):
        for organ in ("Leaves", "Trunks"):
            d = os.path.join(DATA_ROOT, root_kind, split, organ)
            if os.path.isdir(d):
                # remove empty subdirs first
                for root, dirs, files in os.walk(d, topdown=False):
                    if not dirs and not files:
                        try:
                            os.rmdir(root)
                        except OSError:
                            pass
                # if organ now empty, remove it
                try:
                    if not os.listdir(d):
                        os.rmdir(d)
                except Exception:
                    pass


def main():
    # images then labels
    move_under_flat("images", dry_run=False)
    move_under_flat("labels", dry_run=False)
    print(
        "Done: reorganized into flat {genus} {organ} folders under train/val for images and labels.")


if __name__ == "__main__":
    main()
