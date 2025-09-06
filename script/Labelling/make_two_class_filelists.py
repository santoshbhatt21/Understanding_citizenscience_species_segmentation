import argparse
import os
from pathlib import Path
from typing import Optional, Set, List

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Create train/val image filelists containing any of the specified YOLO classes.")
    ap.add_argument("--images-root", required=True, nargs='+',
                    help="One or more images roots (space-separated). Each must contain train/ and val/.")
    ap.add_argument("--labels-root", required=True,
                    help="Dataset labels root (cleaned preferred) e.g. E:/.../DATA_YOLOv8_cleaned/labels")
    ap.add_argument("--classes", default="0,2",
                    help="Comma-separated class ids to keep, e.g. '0,2' for Leaves,Trunks")
    ap.add_argument("--out", default=None,
                    help="Output directory for filelists; default is <images-root>/../filelists")
    ap.add_argument("--yaml-out", default=None,
                    help="Optional path to write a YAML that points train/val to the generated lists")
    return ap.parse_args()


def _mirror_search_in_root(lbl_path: Path, labels_root: Path, images_root: Path) -> Optional[Path]:
    """Try to resolve the image for a label within a single images root."""
    stem = lbl_path.stem
    try:
        rel = lbl_path.relative_to(labels_root)
        rel_no_ext = rel.with_suffix("")
        for ext in IMG_EXTS:
            cand = images_root / rel_no_ext.with_suffix(ext)
            if cand.exists():
                return cand
    except Exception:
        pass

    # Fallback: same folder name
    for ext in IMG_EXTS:
        cand = images_root / lbl_path.parent.name / f"{stem}{ext}"
        if cand.exists():
            return cand

    # Final fallback: recursive name search (slow)
    hits = list(images_root.rglob(f"{stem}*"))
    for h in hits:
        if h.suffix.lower() in IMG_EXTS:
            return h
    return None


def find_image_for_label(lbl_path: Path, labels_root: Path, images_roots: List[Path]) -> Optional[Path]:
    """Search for the matching image across multiple images roots in order."""
    for root in images_roots:
        hit = _mirror_search_in_root(lbl_path, labels_root, root)
        if hit is not None:
            return hit
    return None


def label_has_any_class(lbl_path: Path, wanted: Set[int]) -> bool:
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                except Exception:
                    continue
                if cls in wanted:
                    return True
    except Exception:
        return False
    return False


def build_lists(images_roots: List[Path], labels_root: Path, keep_classes: Set[int], out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    result_paths = {}

    for split in ("train", "val"):
        lbl_split = labels_root / split
        if not lbl_split.exists():
            print(f"Warning: labels split not found -> {lbl_split}")
            continue

        kept = []
        for lbl_path in lbl_split.rglob("*.txt"):
            if label_has_any_class(lbl_path, keep_classes):
                img_path = find_image_for_label(
                    lbl_path, labels_root, images_roots)
                if img_path and img_path.exists():
                    kept.append(img_path.resolve())

        list_path = out_dir / \
            f"{split}_{'-'.join(map(str, sorted(keep_classes)))}.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for p in kept:
                f.write(str(p) + "\n")

        stats[split] = len(kept)
        result_paths[split] = list_path
        print(f"{split}: {len(kept)} images written -> {list_path}")

    return {"stats": stats, "lists": result_paths}


def write_yaml(yaml_out: Path, list_train: Optional[Path], list_val: Optional[Path]):
    yaml_out.parent.mkdir(parents=True, exist_ok=True)
    content = f"""
# Two-class filelists (Leaves=0, Trunks=2) using cleaned labels
train: {list_train if list_train else ''}
val: {list_val if list_val else ''}

# Keep original names; pass classes=[0,2] to the trainer
names:
  0: Leaves
  1: Others
  2: Trunks
""".lstrip()
    yaml_out.write_text(content, encoding="utf-8")
    print(f"YAML written -> {yaml_out}")


def main():
    args = parse_args()
    # Accept multiple images roots, search in order
    images_roots = [Path(p) for p in (args.images_root if isinstance(
        args.images_root, list) else [args.images_root])]
    labels_root = Path(args.labels_root)
    # Default out dir next to the first images root
    out_dir = Path(args.out) if args.out else (
        images_roots[0].parent / "filelists")

    keep = set(int(x) for x in args.classes.split(",") if x.strip() != "")
    print(f"Building filelists for classes: {sorted(keep)}")
    print("Images roots (search order):")
    for r in images_roots:
        print(" -", r)
    print(f"Labels root: {labels_root}")
    res = build_lists(images_roots, labels_root, keep, out_dir)

    if args.yaml_out:
        write_yaml(Path(args.yaml_out), res["lists"].get(
            "train"), res["lists"].get("val"))
    else:
        # default YAML path next to dataset root
        default_yaml = images_roots[0].parent / \
            "data_cleaned_leaves_trunks_lists.yaml"
        write_yaml(default_yaml, res["lists"].get(
            "train"), res["lists"].get("val"))
    print("Done.")


if __name__ == "__main__":
    main()
