import argparse
from pathlib import Path
import cv2
import numpy as np


def _color_for_class(cls: int) -> tuple:
    # Distinct-ish palette; cycles if cls exceeds length
    palette = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 127, 0), (127, 0, 255), (0, 127, 255),
        (127, 255, 0), (255, 0, 127), (0, 255, 127),
    ]
    return palette[cls % len(palette)]


def draw_yolo_seg(image_path: Path, label_path: Path, out_path: Path,
                  alpha: float = 0.4, thickness: int = 2,
                  fill: bool = True) -> bool:
    """
    Draw YOLO-seg polygons from label_path onto image_path and save to out_path.
    Returns True if overlay saved, False if skipped (e.g., missing/empty labels).
    """
    image_path = Path(image_path)
    label_path = Path(label_path)
    out_path = Path(out_path)

    im = cv2.imread(str(image_path))
    if im is None:
        print(f"[WARN] Could not read image: {image_path}")
        return False
    H, W = im.shape[:2]

    if not label_path.exists():
        print(f"[SKIP] Label not found for image: {image_path}")
        return False

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception as e:
        print(f"[WARN] Failed to read label {label_path}: {e}")
        return False

    if not lines:
        print(f"[SKIP] Empty label file: {label_path}")
        return False

    overlay = im.copy()
    drew_any = False

    for ln in lines:
        vals = ln.split()
        if not vals:
            continue
        try:
            cls = int(float(vals[0]))
        except ValueError:
            print(f"[WARN] Bad class id in {label_path}: '{vals[0]}'")
            continue
        nums = vals[1:]
        if len(nums) < 6 or len(nums) % 2 != 0:
            # Need at least 3 points and an even number of coords
            print(f"[WARN] Bad poly length in {label_path}: {len(nums)}")
            continue
        try:
            pts = np.array(list(map(float, nums)),
                           dtype=np.float32).reshape(-1, 2)
        except Exception:
            print(f"[WARN] Non-numeric coords in {label_path}")
            continue

        # denormalize to pixel coords
        pts[:, 0] = np.clip(pts[:, 0] * W, 0, W - 1)
        pts[:, 1] = np.clip(pts[:, 1] * H, 0, H - 1)
        poly = pts.astype(np.int32).reshape(-1, 1, 2)

        color = _color_for_class(cls)
        if fill:
            cv2.fillPoly(overlay, [poly], color)
        cv2.polylines(overlay, [poly], isClosed=True,
                      color=(0, 0, 0), thickness=thickness)
        drew_any = True

    if not drew_any:
        print(f"[SKIP] No valid polygons drawn for {image_path}")
        return False

    vis = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), vis)
    if ok:
        print(f"[OK] Saved: {out_path}")
    else:
        print(f"[ERR] Failed to write: {out_path}")
    return ok


def batch_overlay(images_root: Path, labels_root: Path, out_root: Path,
                  exts=(".jpg", ".jpeg", ".png", ".bmp"),
                  alpha: float = 0.4, thickness: int = 2, fill: bool = True,
                  skip_missing: bool = True) -> dict:
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    out_root = Path(out_root)

    count_total = 0
    count_drawn = 0
    count_missing = 0
    count_empty = 0

    for img_path in images_root.rglob("*"):
        if not img_path.is_file() or img_path.suffix.lower() not in exts:
            continue
        rel = img_path.relative_to(images_root)
        lbl_path = labels_root / rel.with_suffix(".txt")
        out_path = out_root / rel.with_suffix(".jpg")
        count_total += 1

        if not lbl_path.exists():
            count_missing += 1
            if skip_missing:
                print(f"[MISS] {rel}")
                continue

        saved = draw_yolo_seg(img_path, lbl_path, out_path,
                              alpha=alpha, thickness=thickness, fill=fill)
        if saved:
            count_drawn += 1
        else:
            # distinguish empty labels if label file exists but nothing drawn
            if lbl_path.exists():
                count_empty += 1

    return {
        "total_images": count_total,
        "saved_overlays": count_drawn,
        "missing_labels": count_missing,
        "empty_or_bad_labels": count_empty,
    }


def build_argparser():
    p = argparse.ArgumentParser(
        description="Overlay YOLO segmentation labels onto images (single or batch).")
    # Single
    p.add_argument("--image", type=str, help="Path to a single image")
    p.add_argument("--label", type=str,
                   help="Path to corresponding label .txt")
    p.add_argument("--out", type=str, help="Path to save overlay image")

    # Batch
    p.add_argument("--images-root", type=str,
                   help="Root folder of images (will recurse)")
    p.add_argument("--labels-root", type=str,
                   help="Root folder of labels mirroring images-root")
    p.add_argument("--out-root", type=str,
                   help="Root folder to save overlays (mirrors structure)")

    # Options
    p.add_argument("--alpha", type=float, default=0.4,
                   help="Fill transparency (0-1)")
    p.add_argument("--thickness", type=int, default=2,
                   help="Polygon outline thickness")
    p.add_argument("--no-fill", action="store_true",
                   help="Disable filling polygons (outline only)")
    p.add_argument("--no-skip-missing", action="store_true",
                   help="Do not skip images without labels")

    return p


def main():
    args = build_argparser().parse_args()

    fill = not args.no_fill
    skip_missing = not args.no_skip_missing

    # Single
    if args.image and args.label and args.out:
        ok = draw_yolo_seg(Path(args.image), Path(args.label), Path(args.out),
                           alpha=args.alpha, thickness=args.thickness, fill=fill)
        if not ok:
            raise SystemExit(1)
        return

    # Batch
    if args.images_root and args.labels_root and args.out_root:
        stats = batch_overlay(Path(args.images_root), Path(args.labels_root), Path(args.out_root),
                              alpha=args.alpha, thickness=args.thickness, fill=fill,
                              skip_missing=skip_missing)
        print("\nSummary:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        # non-zero empty or missing considered success but noteworthy
        return

    print("Please provide either --image/--label/--out for single, or --images-root/--labels-root/--out-root for batch.")
    print("Use --help for details.")


if __name__ == "__main__":
    main()
