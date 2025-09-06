import os
import glob
import argparse
from pathlib import Path

import cv2
import numpy as np

# ---------- heuristics (tune if needed) ----------
# Note: these can be overridden from the CLI (see --help)
MIN_VERTICES = 6            # need at least 3 points (=6 numbers)
MIN_AREA_RATIO = 0.001      # polygon area >= 0.1% of image area
MIN_WH_PX = 12              # bbox width AND height >= 12 px
# points within this many pixels of the border are considered "on border"
BORDER_PX = 2
BORDER_POINT_FRAC = 0.50    # drop mask if >=50% of vertices lie on the border
# Optional upper bound (drop if polygon area > MAX_AREA_RATIO * image area). None disables.
MAX_AREA_RATIO = None

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def shoelace_area(pts):
    """Polygon area in px^2. pts: (N,2) array of (x,y) pixels."""
    if pts.shape[0] < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def load_image_size(img_path):
    im = cv2.imread(img_path)
    if im is None:
        return None
    h, w = im.shape[:2]
    return (w, h)


def parse_line(line):
    parts = line.strip().split()
    if len(parts) < 1:
        return None, None
    try:
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
    except ValueError:
        return None, None
    return cls, coords


def norm_to_px(coords, W, H):
    """coords [x1,y1,x2,y2,...] normalized -> (N,2) pixels clipped."""
    arr = np.array(coords, dtype=np.float32).reshape(-1, 2)
    arr[:, 0] = np.clip(arr[:, 0], 0, 1) * W
    arr[:, 1] = np.clip(arr[:, 1], 0, 1) * H
    return arr


def is_degenerate_polygon(px_pts, W, H):
    """Return True if polygon is tiny, flat, border-hugging or invalid."""
    # invalid sizes
    if px_pts.shape[0] < 3:
        return True

    # bbox filter
    x1, y1 = px_pts.min(axis=0)
    x2, y2 = px_pts.max(axis=0)
    bw, bh = (x2 - x1), (y2 - y1)
    if bw < MIN_WH_PX or bh < MIN_WH_PX:
        return True

    # area filter (relative to image)
    area = shoelace_area(px_pts)
    if area < (W * H * MIN_AREA_RATIO):
        return True
    if MAX_AREA_RATIO is not None and area > (W * H * float(MAX_AREA_RATIO)):
        return True

    # border-hugging filter
    on_border = (
        (px_pts[:, 0] <= BORDER_PX) |
        (px_pts[:, 1] <= BORDER_PX) |
        (px_pts[:, 0] >= (W - 1 - BORDER_PX)) |
        (px_pts[:, 1] >= (H - 1 - BORDER_PX))
    )
    if on_border.mean() >= BORDER_POINT_FRAC:
        return True

    return False


def find_image_for_label(label_path: str, labels_root: str, images_root: str):
    """Find matching image for a label by mirroring the relative path under images_root.

    Falls back to a recursive name search if the mirrored path is missing.
    """
    lp = Path(label_path)
    lr = Path(labels_root)
    ir = Path(images_root)
    stem = lp.stem

    # Try mirrored relative path first (handles nested folders)
    try:
        rel = lp.relative_to(lr)
        rel_no_ext = rel.with_suffix("")
        for ext in IMG_EXTS:
            p = ir / rel_no_ext.with_suffix(ext)
            if p.exists():
                return str(p)
    except ValueError:
        # label not under labels_root; skip to fallback
        pass

    # Fallback: same immediate parent folder name under images_root
    for ext in IMG_EXTS:
        p = ir / lp.parent.name / f"{stem}{ext}"
        if p.exists():
            return str(p)

    # Final fallback: recursive search by filename
    for ext in IMG_EXTS:
        hits = list(ir.rglob(f"{stem}{ext}"))
        if hits:
            return str(hits[0])
    return None


def process_one_label(lbl_path, images_root, labels_root, out_root):
    img_path = find_image_for_label(lbl_path, labels_root, images_root)
    if not img_path:
        return {"kept": 0, "removed": 0, "empty": False, "reason": "image_not_found"}

    size = load_image_size(img_path)
    if size is None:
        return {"kept": 0, "removed": 0, "empty": False, "reason": "image_broken"}
    W, H = size

    kept_lines = []
    removed = 0

    with open(lbl_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            cls, coords = parse_line(raw)
            if cls is None or coords is None:
                removed += 1
                continue
            if len(coords) < MIN_VERTICES or (len(coords) % 2) != 0:
                removed += 1
                continue

            px = norm_to_px(coords, W, H)
            if is_degenerate_polygon(px, W, H):
                removed += 1
                continue

            kept_lines.append(
                f"{cls} " + " ".join(f"{v:.6f}" for v in np.clip(np.array(coords), 0, 1)))

    # write to out path mirroring the structure
    rel = Path(lbl_path).relative_to(out_root["labels_in_root"])
    out_lbl = Path(out_root["labels_out_root"]) / rel
    out_lbl.parent.mkdir(parents=True, exist_ok=True)

    if kept_lines:
        with open(out_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(kept_lines) + "\n")
        return {"kept": len(kept_lines), "removed": removed, "empty": False, "reason": "ok"}
    else:
        # write an empty file to mark 'no objects' (YOLO expects either empty file or no file)
        open(out_lbl, "w", encoding="utf-8").close()
        return {"kept": 0, "removed": removed, "empty": True, "reason": "emptied"}


def main():
    ap = argparse.ArgumentParser(
        description="Clean tiny/degenerate YOLO segmentation polygons.")
    ap.add_argument("--data", required=False,
                    help="Dataset root containing images/ and labels/ (with train/val/...)")
    ap.add_argument("--images", required=False,
                    help="Explicit images root (overrides --data/images)")
    ap.add_argument("--labels", required=False,
                    help="Explicit labels root (overrides --data/labels)")
    ap.add_argument("--out", required=True,
                    help="Output dataset root for cleaned labels (mirrors labels/ structure).")
    # Optional threshold overrides
    ap.add_argument("--min-area-ratio", type=float, default=None,
                    help="Drop polygons with area < r * image_area (default 0.001)")
    ap.add_argument("--min-wh-px", type=int, default=None,
                    help="Drop polygons whose bbox width or height < px (default 12)")
    ap.add_argument("--border-px", type=int, default=None,
                    help="A vertex within this many pixels from an edge counts as on-border (default 2)")
    ap.add_argument("--border-frac", type=float, default=None,
                    help="Drop if >= this fraction of vertices are on-border (default 0.50)")
    ap.add_argument("--max-area-ratio", type=float, default=None,
                    help="Optional: drop polygons with area > r * image_area (disabled by default)")
    args = ap.parse_args()

    if not args.data and not (args.images and args.labels):
        raise SystemExit(
            "Provide either --data or both --images and --labels.")

    # Apply CLI overrides to module-level thresholds
    global MIN_AREA_RATIO, MIN_WH_PX, BORDER_PX, BORDER_POINT_FRAC, MAX_AREA_RATIO
    if args.min_area_ratio is not None:
        MIN_AREA_RATIO = float(args.min_area_ratio)
    if args.min_wh_px is not None:
        MIN_WH_PX = int(args.min_wh_px)
    if args.border_px is not None:
        BORDER_PX = int(args.border_px)
    if args.border_frac is not None:
        BORDER_POINT_FRAC = float(args.border_frac)
    if args.max_area_ratio is not None:
        MAX_AREA_RATIO = float(args.max_area_ratio)

    data_root = Path(args.data) if args.data else None
    images_root = Path(args.images) if args.images else (data_root / "images")
    labels_in_root = Path(args.labels) if args.labels else (
        data_root / "labels")
    out_root = Path(args.out)
    labels_out_root = out_root / "labels"

    if not labels_in_root.exists() or not images_root.exists():
        raise SystemExit(
            "Could not find images/ or labels/ root. Check --data or --images/--labels paths.")

    stats = {"files": 0, "polys_kept": 0,
             "polys_removed": 0, "emptied_files": 0}

    # pass roots to process_one_label
    payload = {"labels_in_root": str(
        labels_in_root), "labels_out_root": str(labels_out_root)}

    label_files = []
    for split in ["train", "val", "valid", "test"]:
        label_files += glob.glob(str(labels_in_root /
                                 split / "**/*.txt"), recursive=True)
    if not label_files:
        label_files = glob.glob(
            str(labels_in_root / "**/*.txt"), recursive=True)

    for lp in label_files:
        res = process_one_label(lp, str(images_root),
                                str(labels_in_root), payload)
        stats["files"] += 1
        stats["polys_kept"] += res["kept"]
        stats["polys_removed"] += res["removed"]
        if res["empty"]:
            stats["emptied_files"] += 1

    # copy nothing else; you will still point YOLO to the same images/, but labels= new out path
    print("\n=== CLEAN SUMMARY ===")
    for k, v in stats.items():
        print(f"{k:>14}: {v}")
    print(f"\nClean labels written under: {labels_out_root}")
    print("Use the same images/, but set data.yaml 'labels' to this new path or swap labels folders.")
    print("\nTip: re-run your debug overlay on a few samples to verify masks look sensible.")
    print("\nThresholds used -> min_area_ratio=%.5f, min_wh_px=%d, border_px=%d, border_frac=%.2f, max_area_ratio=%s"
          % (MIN_AREA_RATIO, MIN_WH_PX, BORDER_PX, BORDER_POINT_FRAC,
             ("None" if MAX_AREA_RATIO is None else f"{MAX_AREA_RATIO:.2f}")))


if __name__ == "__main__":
    main()
# End of clean_yolo_polygon.py
