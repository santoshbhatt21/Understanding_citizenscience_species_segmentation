import os
import csv
import shutil
import argparse
import re
from collections import Counter, defaultdict
from typing import List, Dict, Optional

# =========================
# Defaults for no-CLI usage (set these manually if you prefer)
# Leave as None/"" to disable a default.
DEFAULT_CSV = r"E:/Santosh_master_thesis/prediction_metadata_LOT_10_species.csv"
DEFAULT_META_ROOT = None  # e.g., r"E:/Santosh_master_thesis/metadata_folder"
DEFAULT_OUT = r"E:/Santosh_master_thesis/LT_species_organ_10_species"
DEFAULT_CLASSES = "Leaves,Trunks"
DEFAULT_LIMIT = 0


def parse_args():
    p = argparse.ArgumentParser(
        description="Copy images into hierarchical class/species folders (Leaves/Trunks/species) or flat 20 species-organ classes from prediction metadata.")
    g = p.add_mutually_exclusive_group(required=False)
    g.add_argument(
        "--csv", help="Path to a single predictions CSV (with columns: image_path,predicted_class,confidence)")
    g.add_argument(
        "--meta-root", help="Root directory containing one or more prediction CSV files (we'll scan *.csv)")
    p.add_argument("--out", default=DEFAULT_OUT or "./LOT_flat",
                   help="Output root directory. By default creates '<Genus species> leaves|trunks' (20 classes). Use --hierarchical for Leaves/<species>/ and Trunks/<species>/.")
    p.add_argument("--classes", default=DEFAULT_CLASSES,
                   help="Comma-separated classes to include (default: Leaves,Trunks). Others is excluded by default.")
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                   help="Optional max images per class (0 = no limit)")
    # Default to flat 20-class layout; allow override to hierarchical
    p.add_argument("--flat-20-classes", dest="flat_20_classes", action="store_true", default=True,
                   help="Output 20 species-organ classes like 'Abies alba leaves', 'Abies alba trunks' (default).")
    p.add_argument("--hierarchical", dest="flat_20_classes", action="store_false",
                   help="Use hierarchical layout: Leaves/<species>/ and Trunks/<species>/.")
    return p.parse_args()


def load_rows_from_csv(csv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Basic header check
            if not set(["image_path", "predicted_class"]).issubset(reader.fieldnames or []):
                return rows
            for row in reader:
                rows.append(row)
    except Exception:
        pass
    return rows


def find_csvs_under_root(root_dir: str) -> List[str]:
    out: List[str] = []
    for fname in os.listdir(root_dir):
        if not fname.lower().endswith(".csv"):
            continue
        out.append(os.path.join(root_dir, fname))
    return out


def main():
    args = parse_args()
    include_classes = [c.strip()
                       for c in (args.classes or "").split(",") if c.strip()]
    if not include_classes:
        include_classes = ["Leaves", "Trunks"]

    os.makedirs(args.out, exist_ok=True)
    print(f"[Mode] {'Flat 20 species-organ classes' if args.flat_20_classes else 'Hierarchical Leaves/Trunks/species'}")

    # Load metadata rows from either a single CSV or scan a root dir
    rows: List[Dict[str, str]] = []

    # Resolve sources: CLI > defaults
    resolved_csv = args.csv if args.csv else (
        DEFAULT_CSV if DEFAULT_CSV else None)
    resolved_root = args.meta_root if args.meta_root else (
        DEFAULT_META_ROOT if DEFAULT_META_ROOT else None)

    if resolved_csv and os.path.isfile(resolved_csv):
        if not args.csv:
            print(f"[Info] Using DEFAULT_CSV: {resolved_csv}")
        args.csv = resolved_csv
        rows.extend(load_rows_from_csv(args.csv))
    elif resolved_root and os.path.isdir(resolved_root):
        if not args.meta_root:
            print(f"[Info] Using DEFAULT_META_ROOT: {resolved_root}")
        args.meta_root = resolved_root
        for csv_path in find_csvs_under_root(args.meta_root):
            rows.extend(load_rows_from_csv(csv_path))
    else:
        raise SystemExit(
            "Provide either --csv <file> or --meta-root <dir> with CSVs, or set DEFAULT_CSV/DEFAULT_META_ROOT in the script.")

    counts = Counter()  # counts per destination class (varies by mode)
    # dest class -> species -> count
    per_species = defaultdict(lambda: Counter())
    errors = 0
    seen = set()

    def infer_species_folder(path: str) -> str:
        return os.path.basename(os.path.dirname(path))

    def to_binomial(species_folder: str) -> str:
        """Convert folder like '001_Abies_alba' or 'Abies_alba' to 'Abies alba'."""
        name = re.sub(r"^\d+_+", "", species_folder)
        parts = re.split(r"[_\s]+", name)
        if len(parts) >= 2:
            genus = parts[0].capitalize()
            species = parts[1].lower()
            return f"{genus} {species}"
        return parts[0].capitalize() if parts else species_folder

    for row in rows:
        img_path: Optional[str] = row.get("image_path")
        pred_class: Optional[str] = row.get("predicted_class")
        if not img_path or not pred_class:
            continue
        if pred_class not in include_classes:
            continue  # exclude Other classes by default
        if img_path in seen:
            continue

        # Species inferred from parent directory name (…/species/image.jpg)
        species = infer_species_folder(img_path)

        # Decide destination dir based on mode
        if args.flat_20_classes:
            organ_suffix = "leaves" if pred_class.lower().startswith("leave") else "trunks"
            dest_class = f"{to_binomial(species)} {organ_suffix}"
            dst_dir = os.path.join(args.out, dest_class)
        else:
            dest_class = pred_class
            dst_dir = os.path.join(args.out, pred_class, species)

        # Enforce optional per-class limit (applies to destination class)
        if args.limit and counts[dest_class] >= args.limit:
            continue

        os.makedirs(dst_dir, exist_ok=True)

        try:
            shutil.copy(img_path, dst_dir)
            counts[dest_class] += 1
            per_species[dest_class][species] += 1
            seen.add(img_path)
        except Exception:
            errors += 1
            # keep going

    # Summary
    print("\nSummary (images copied):")
    total = 0
    # Summary: list destination classes we actually wrote
    for cls in sorted(counts.keys()):
        cls_total = counts[cls]
        print(f"  {cls}: {cls_total}")
        for sp, n in sorted(per_species[cls].items()):
            print(f"    - {sp}: {n}")
        total += cls_total
    print(f"Total copied: {total}")
    if errors:
        print(f"Warnings: {errors} copy errors (missing/locked files, etc.)")


if __name__ == "__main__":
    main()
