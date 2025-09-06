import os
import glob
from PIL import Image


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# Config
DATA_ROOT = r"E:/Santosh_master_thesis/DATA_YOLOv8"
MIN_AREA_PX = 64          # drop polygons smaller than this area
MIN_VERTICES = 6          # drop polygons with fewer vertices
BACKUP_DIR = r"./annotation_probe_outputs/backup_labels"


def norm_to_abs(coords, W, H):
    pts = []
    for i in range(0, len(coords), 2):
        x = max(0, min(W - 1, coords[i] * W))
        y = max(0, min(H - 1, coords[i + 1] * H))
        pts.append((x, y))
    return pts


def polygon_area(pts):
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def process_split(images_dir, labels_dir):
    print(f"[CLEAN] Processing split: {images_dir} | {labels_dir}")
    # recurse through nested label folders
    txts = glob.glob(os.path.join(labels_dir, "**", "*.txt"), recursive=True)
    kept, dropped_polys, empty_files = 0, 0, 0
    ensure_dir(BACKUP_DIR)

    for lp in txts:
        # map image by mirroring relative path from labels->images
        rel = os.path.relpath(lp, labels_dir)
        rel_img_base = os.path.splitext(rel)[0]
        ip = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = os.path.join(images_dir, rel_img_base + ext)
            if os.path.exists(p):
                ip = p
                break
        if not ip:
            continue
        try:
            W, H = Image.open(ip).size
        except Exception:
            continue

        with open(lp, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

        out_lines = []
        for ln in lines:
            parts = ln.split()
            if len(parts) > 5 and len(parts) % 2 == 1:
                cls = parts[0]
                coords = list(map(float, parts[1:]))
                pts = norm_to_abs(coords, W, H)
                area = polygon_area(pts)
                if area < MIN_AREA_PX or len(pts) < MIN_VERTICES:
                    dropped_polys += 1
                    continue
                out_lines.append(ln)
            else:
                # keep non-polygon lines (e.g., bbox) unchanged
                out_lines.append(ln)

        if not out_lines:
            empty_files += 1
        else:
            kept += 1

        # backup and write
        try:
            # backup preserving subfolder structure
            bkp = os.path.join(BACKUP_DIR, os.path.relpath(lp, labels_dir))
            ensure_dir(os.path.dirname(bkp))
            if not os.path.exists(bkp):
                with open(bkp, "w", encoding="utf-8") as bf:
                    bf.write("\n".join(lines) + "\n")
            # write cleaned
            ensure_dir(os.path.dirname(lp))
            with open(lp, "w", encoding="utf-8") as f:
                if out_lines:
                    f.write("\n".join(out_lines) + "\n")
                else:
                    # leave a truly empty label file (Ultralytics handles as no targets)
                    f.write("")
        except Exception as e:
            print(f"[CLEAN] Failed to write {lp}: {e}")

    print(
        f"[CLEAN] Done. kept_files={kept}, dropped_polys={dropped_polys}, emptied_files={empty_files}")


def main():
    splits = [
        (os.path.join(DATA_ROOT, "images", "train"),
         os.path.join(DATA_ROOT, "labels", "train")),
        (os.path.join(DATA_ROOT, "images", "val"),
         os.path.join(DATA_ROOT, "labels", "val")),
    ]
    for imgs, labs in splits:
        if os.path.isdir(imgs) and os.path.isdir(labs):
            process_split(imgs, labs)


if __name__ == "__main__":
    main()
