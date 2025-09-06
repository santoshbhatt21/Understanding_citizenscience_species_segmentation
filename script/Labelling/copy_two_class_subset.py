import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Materialize a two-class YOLO dataset (copy images and filtered labels) from existing dataset.")
    ap.add_argument("--images-root", required=True, nargs='+',
                    help="One or more images roots (space-separated). Each contains train/ and val/.")
    ap.add_argument("--labels-root", required=True,
                    help="Labels root (contains train/ and val/). Typically the CLEANED labels root.")
    ap.add_argument("--out", required=True,
                    help="Output dataset root to create (will have images/ and labels/ subfolders).")
    ap.add_argument("--classes", default="0,2",
                    help="Comma-separated original class IDs to keep (e.g., '0,2' for Leaves,Trunks).")
    ap.add_argument("--names", default="Leaves,Trunks",
                    help="Comma-separated class names for the new dataset (same order as remapped).")
    ap.add_argument("--preserve-ids", action="store_true",
                    help="Do NOT remap class IDs; keep original IDs in labels (not recommended unless IDs are contiguous from 0..nc-1).")
    ap.add_argument("--include-negatives", action="store_true",
                    help="Copy images even if, after filtering, they have zero kept objects (labels will be empty).")
    return ap.parse_args()


def _mirror_search(lbl_path: Path, labels_root: Path, images_root: Path) -> Optional[Path]:
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
    # same folder name
    for ext in IMG_EXTS:
        cand = images_root / lbl_path.parent.name / f"{stem}{ext}"
        if cand.exists():
            return cand
    # recursive name search
    for ext in IMG_EXTS:
        hits = list(images_root.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def find_image_for_label(lbl_path: Path, labels_root: Path, images_roots: List[Path]) -> Optional[Path]:
    for root in images_roots:
        p = _mirror_search(lbl_path, labels_root, root)
        if p is not None:
            return p
    return None


def filter_label_lines(lbl_path: Path, keep: Set[int], remap: Dict[int, int]) -> List[str]:
    kept: List[str] = []
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
                if cls in keep:
                    new_cls = remap.get(cls, cls)
                    parts[0] = str(new_cls)
                    kept.append(" ".join(parts))
    except Exception:
        return []
    return kept


def copy_subset(images_roots: List[Path], labels_root: Path, out_root: Path, keep_ids: List[int], names: List[str],
                preserve_ids: bool, include_negatives: bool) -> Dict[str, int]:
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    # mapping original id -> new id
    keep_set: Set[int] = set(keep_ids)
    remap: Dict[int, int] = {k: (k if preserve_ids else i)
                             for i, k in enumerate(keep_ids)}

    stats = {"kept_images": 0, "kept_labels": 0,
             "skipped_no_target": 0, "missing_images": 0}

    for split in ("train", "val"):
        in_split = labels_root / split
        if not in_split.exists():
            print(f"Warning: labels split not found -> {in_split}")
            continue

        for lbl_path in in_split.rglob("*.txt"):
            img_src = find_image_for_label(lbl_path, labels_root, images_roots)
            if img_src is None or not img_src.exists():
                stats["missing_images"] += 1
                continue

            lines = filter_label_lines(lbl_path, keep_set, remap)
            if not lines and not include_negatives:
                stats["skipped_no_target"] += 1
                continue

            # derive relative path from labels_root for consistent structure
            rel_lbl = lbl_path.relative_to(labels_root)
            out_lbl = out_labels / rel_lbl
            out_img = out_images / rel_lbl.with_suffix(img_src.suffix)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            out_img.parent.mkdir(parents=True, exist_ok=True)

            # write label (empty file if negatives included with no targets)
            with open(out_lbl, "w", encoding="utf-8") as f:
                if lines:
                    f.write("\n".join(lines) + "\n")
                else:
                    pass
            shutil.copy2(str(img_src), str(out_img))
            stats["kept_images"] += 1
            stats["kept_labels"] += 1

    # write dataset YAML
    names_map = {i: n for i, n in enumerate(names)}
    yaml_text = [
        f"path: {out_root.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ] + [f"  {i}: {n}" for i, n in names_map.items()]
    (out_root / "data.yaml").write_text("\n".join(yaml_text) + "\n", encoding="utf-8")

    return stats


def main():
    args = parse_args()
    images_roots = [Path(p) for p in args.images_root]
    labels_root = Path(args.labels_root)
    out_root = Path(args.out)
    keep_ids = [int(x) for x in args.classes.split(",") if x.strip() != ""]
    names = [s.strip() for s in args.names.split(",") if s.strip() != ""]
    if not args.preserve_ids and len(names) != len(keep_ids):
        print("Warning: names count does not match kept classes; using generic names.")
        names = [f"class_{i}" for i in range(len(keep_ids))]

    print("Images roots (search order):")
    for r in images_roots:
        print(" -", r)
    print("Labels root:", labels_root)
    print("Output root:", out_root)
    print("Keeping class IDs:", keep_ids, " Remap:",
          ("preserve" if args.preserve_ids else "to 0..k-1"))

    stats = copy_subset(images_roots, labels_root, out_root,
                        keep_ids, names, args.preserve_ids, args.include_negatives)
    print("\n=== COPY SUMMARY ===")
    for k, v in stats.items():
        print(f"{k:>18}: {v}")
    print("Dataset YAML:", (out_root / "data.yaml"))


if __name__ == "__main__":
    main()
