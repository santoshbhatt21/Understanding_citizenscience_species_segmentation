import argparse
from pathlib import Path
from typing import List, Optional, Dict
import sys


def load_classes_from_file(path: Optional[Path]) -> List[str]:
    if not path:
        return []
    out: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            # Allow formats: "name" or "id name" or "id,name"
            if "," in s:
                parts = [p.strip() for p in s.split(",", 1)]
            else:
                parts = s.split()
            if parts and parts[0].isdigit() and len(parts) >= 2:
                name = " ".join(parts[1:])
            else:
                name = s
            out.append(name)
    return out


def discover_classes(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def find_class_dir(base: Path, class_name: str, strip_suffix: str) -> Optional[Path]:
    # 1) try exact match
    p = base / class_name
    if p.is_dir():
        return p
    # 2) try stripping suffix
    if strip_suffix and class_name.endswith(strip_suffix):
        alt = base / class_name[: -len(strip_suffix)]
        if alt.is_dir():
            return alt
    # 3) try adding suffix
    if strip_suffix and not class_name.endswith(strip_suffix):
        alt2 = base / f"{class_name}{strip_suffix}"
        if alt2.is_dir():
            return alt2
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Visualize YOLO-seg overlays for all classes.")
    ap.add_argument("--images-root", required=True,
                    help="Root images folder with per-class subfolders")
    ap.add_argument("--labels-root", required=True,
                    help="Root labels folder with per-class subfolders")
    ap.add_argument("--out-root", required=True,
                    help="Where to save overlays (mirrors per-class structure)")
    ap.add_argument(
        "--classes-file", help="Optional file listing class folder names (one per line)")
    ap.add_argument("--strip-suffix", default="_mask",
                    help="Suffix to map between class folder names (images/labels)")
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--no-fill", action="store_true")
    ap.add_argument("--no-skip-missing", action="store_true")

    args = ap.parse_args()

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    classes = load_classes_from_file(
        Path(args.classes_file) if args.classes_file else None)
    if not classes:
        # Prefer discovering from labels_root (since labels were just created), fallback to images_root
        classes = discover_classes(
            labels_root) or discover_classes(images_root)

    if not classes:
        print(
            "[ERR] No classes found. Provide --classes-file or ensure class subfolders exist.")
        raise SystemExit(2)

    # Import batch_overlay from the sibling module
    this_dir = Path(__file__).parent
    sys.path.insert(0, str(this_dir))
    try:
        from visualize_yolo_seg_labels import batch_overlay
    finally:
        if str(this_dir) in sys.path:
            sys.path.remove(str(this_dir))

    fill = not args.no_fill
    skip_missing = not args.no_skip_missing

    totals: Dict[str, int] = {
        "total_images": 0, "saved_overlays": 0, "missing_labels": 0, "empty_or_bad_labels": 0}

    for cname in classes:
        img_dir = find_class_dir(images_root, cname, args.strip_suffix)
        lbl_dir = find_class_dir(labels_root, cname, args.strip_suffix)
        if not img_dir:
            print(
                f"[SKIP] Images dir not found for class '{cname}' under {images_root}")
            continue
        if not lbl_dir:
            print(
                f"[SKIP] Labels dir not found for class '{cname}' under {labels_root}")
            continue

        out_dir = out_root / (cname[:-len(args.strip_suffix)]
                              if args.strip_suffix and cname.endswith(args.strip_suffix) else cname)
        stats = batch_overlay(img_dir, lbl_dir, out_dir, alpha=args.alpha,
                              thickness=args.thickness, fill=fill, skip_missing=skip_missing)
        print(f"CLASS: {cname} -> {stats}")
        for k in totals:
            totals[k] += stats.get(k, 0)

    print("\nSummary:")
    for k, v in totals.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
