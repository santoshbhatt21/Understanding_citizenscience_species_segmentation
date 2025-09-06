import os
import re
from typing import Dict, Tuple


DATA_ROOT = r"E:\Santosh_master_thesis\DATA_YOLO11_LT"
SOURCE_SPECIES_ROOTS = [
    r"E:\Santosh_master_thesis\LT_images_from_meta_10_species\Leaves",
    r"E:\Santosh_master_thesis\LT_images_from_meta_10_species\Trunks",
]


def parse_species_binomial(folder_name: str) -> Tuple[str, str]:
    """
    Convert '001_Abies_alba' -> ('abies', 'Abies alba').
    Heuristics: strip leading digits/underscores, split by '_' and take first two tokens.
    """
    name = re.sub(r"^\d+_+", "", folder_name)
    parts = re.split(r"[_\s]+", name)
    if len(parts) < 2:
        # Fallback: treat entire as genus, empty species
        genus = parts[0] if parts else folder_name
        binomial = genus.capitalize()
        return genus.lower(), binomial
    genus_raw, species_raw = parts[0], parts[1]
    genus = genus_raw.capitalize()
    species = species_raw.lower()
    return genus.lower(), f"{genus} {species}"


def build_genus_to_binomial_map() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for root in SOURCE_SPECIES_ROOTS:
        if not os.path.isdir(root):
            continue
        for entry in os.listdir(root):
            path = os.path.join(root, entry)
            if not os.path.isdir(path):
                continue
            genus_key, binomial = parse_species_binomial(entry)
            # Don't overwrite an existing mapping with a different binomial
            if genus_key not in mapping:
                mapping[genus_key] = binomial
    return mapping


def rename_genus_dirs(mapping: Dict[str, str]) -> int:
    renames = 0
    pattern = re.compile(
        r"^(?P<genus>[A-Za-z]+)\s+(?P<organ>leaves|trunk|trunks)$", re.IGNORECASE)
    for root_kind in ("images", "labels"):
        for split in ("train", "val"):
            split_dir = os.path.join(DATA_ROOT, root_kind, split)
            if not os.path.isdir(split_dir):
                continue
            for name in list(os.listdir(split_dir)):
                src = os.path.join(split_dir, name)
                if not os.path.isdir(src):
                    continue
                m = pattern.match(name)
                if not m:
                    # Skip non-flat or already species-level dirs
                    continue
                genus = m.group('genus').lower()
                organ = m.group('organ').lower()
                if genus not in mapping:
                    print(
                        f"[WARN] No species mapping for genus '{genus}' (skipping {src})")
                    continue
                binomial = mapping[genus]
                organ_suffix = 'leaves' if organ.startswith(
                    'leave') else 'trunks'
                new_name = f"{binomial} {organ_suffix}"
                dst = os.path.join(split_dir, new_name)
                if os.path.normcase(src) == os.path.normcase(dst):
                    continue
                # If destination exists (from the other root_kind), ensure we don't conflict here
                i = 1
                final_dst = dst
                while os.path.exists(final_dst):
                    # If destination exists and is intended for the same rename, it's okay for images/labels separately
                    # But avoid accidental collision with a different dir by appending a numeric suffix
                    if os.path.isdir(final_dst) and name.lower() == new_name.lower():
                        break
                    final_dst = f"{dst}_{i}"
                    i += 1
                os.replace(src, final_dst)
                renames += 1
                print(f"Renamed: {src} -> {final_dst}")
    return renames


def main():
    mapping = build_genus_to_binomial_map()
    if not mapping:
        print("No genus->species mapping found. Ensure SOURCE_SPECIES_ROOTS exist.")
        return
    n = rename_genus_dirs(mapping)
    print(
        f"Done: {n} folders renamed to species binomial names (with leaves/trunks).")


if __name__ == "__main__":
    main()
