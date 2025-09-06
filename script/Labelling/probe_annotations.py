import os
import re
import glob
import json
import random
import math
import traceback
from collections import Counter, defaultdict

from PIL import Image, ImageDraw
import numpy as np

# Optional but nice to have for COCO RLE masks (we handle polygons without it)
try:
    from pycocotools import mask as coco_mask
    HAS_COCO = True
except Exception:
    HAS_COCO = False

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False


# =========================
# Config
# =========================
DATA_ROOT = r"E:/Santosh_master_thesis/DATA_YOLOv8"      # <-- CHANGE THIS
OUT_DIR = r"./annotation_probe_outputs"      # where outputs go
N_VIS = 12                                  # number of visualization samples

# Tiny-polygon heuristics (in pixel area at image resolution)
TINY_POLY_AREA_PX = 64        # polygons smaller than this will be counted as tiny
MIN_POLY_VERTICES = 6         # polygons with fewer vertices are considered weak/noisy


# =========================
# Helpers
# =========================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def find_first(patterns):
    for p in patterns:
        matches = glob.glob(p, recursive=True)
        if matches:
            return matches
    return []


def guess_yolo_sets(root):
    """
    Try to locate YOLO 'images/*' and 'labels/*' structure.
    Returns [(images_dir, labels_dir, split_name), ...]
    """
    candidates = []
    for split in ["train", "val", "valid", "val2017", "test"]:
        imgs = find_first([os.path.join(root, f"images/{split}"),
                           os.path.join(root, f"images\\{split}")])
        labs = find_first([os.path.join(root, f"labels/{split}"),
                           os.path.join(root, f"labels\\{split}")])
        if imgs and labs and os.path.isdir(imgs[0]) and os.path.isdir(labs[0]):
            candidates.append((imgs[0], labs[0], split))
    # fallback: try any images/ + labels/
    if not candidates:
        imgs = find_first([os.path.join(root, "images"),
                          os.path.join(root, "images/*")])
        labs = find_first([os.path.join(root, "labels"),
                          os.path.join(root, "labels/*")])
        if imgs and labs:
            # pair first matching depths
            for i in imgs:
                # find sibling labels folder at same depth
                base = os.path.basename(i)
                maybe = os.path.join(os.path.dirname(
                    os.path.dirname(i)), "labels", base)
                if os.path.isdir(maybe):
                    candidates.append((i, maybe, base))
    return candidates


def load_yaml_names(root):
    """
    Try to find a YOLO data.yaml and return names mapping.
    """
    yaml_paths = find_first([os.path.join(root, "data.yaml"),
                             os.path.join(root, "**/data.yaml"),
                             os.path.join(root, "*.yaml")])
    names = None
    if yaml_paths:
        path = yaml_paths[0]
        try:
            if HAS_YAML:
                with open(path, "r", encoding="utf-8") as f:
                    y = yaml.safe_load(f)
                # Ultralytics formats: names: {0: 'a', 1: 'b', ...} or list
                n = y.get("names")
                if isinstance(n, dict):
                    # Ensure order by key
                    names = [n[k]
                             for k in sorted(n.keys(), key=lambda x: int(x))]
                elif isinstance(n, list):
                    names = n
            else:
                # minimal parser
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                m = re.search(r"names\s*:\s*\[(.*?)\]", text, flags=re.S)
                if m:
                    raw = m.group(1)
                    names = [t.strip().strip("'\"") for t in raw.split(",")]
        except Exception:
            pass
    return names


def norm_to_abs_poly(poly_norm, W, H):
    # poly_norm: [x1, y1, x2, y2, ...] normalized 0..1
    pts = []
    for i in range(0, len(poly_norm), 2):
        x = max(0, min(W - 1, poly_norm[i] * W))
        y = max(0, min(H - 1, poly_norm[i + 1] * H))
        pts.append((x, y))
    return pts


def polygon_area(pts):
    """Shoelace formula for polygon area; pts: [(x,y),...]."""
    if len(pts) < 3:
        return 0.0
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5


def yolo_bbox_norm_to_xyxy(xc, yc, w, h, W, H):
    xc *= W
    yc *= H
    w *= W
    h *= H
    x1 = max(0, xc - w/2)
    y1 = max(0, yc - h/2)
    x2 = min(W-1, xc + w/2)
    y2 = min(H-1, yc + h/2)
    return x1, y1, x2, y2


def draw_vis(pil_img, anns, class_names, out_path):
    """
    anns: list of dicts with keys:
      - 'cls' (int)
      - either 'bbox' (x1,y1,x2,y2) or 'poly' (list of (x,y))
    """
    img = pil_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for a in anns:
        cname = class_names[a['cls']] if class_names and a['cls'] < len(
            class_names) else str(a['cls'])
        color = (0, 255, 0, 128)
        if 'bbox' in a:
            x1, y1, x2, y2 = a['bbox']
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0, 255), width=3)
            draw.text((x1+2, y1+2), cname, fill=(255, 255, 255, 255))
        if 'poly' in a:
            pts = a['poly']
            draw.polygon(pts, outline=(255, 0, 0, 255))
            # label near first point
            if pts:
                draw.text((pts[0][0]+2, pts[0][1]+2),
                          cname, fill=(255, 255, 255, 255))
    ensure_dir(os.path.dirname(out_path))
    img.save(out_path)


# =========================
# Probing functions
# =========================
def probe_yolo(data_root):
    sets = guess_yolo_sets(data_root)
    if not sets:
        return None
    print(
        f"[YOLO] Found image/label sets: {[(os.path.basename(a), os.path.basename(b), s) for a,b,s in sets]}")
    names = load_yaml_names(data_root)
    if names:
        print(f"[YOLO] Class names from data.yaml: {names}")
    else:
        print("[YOLO] Could not find data.yaml names — will show numeric IDs.")

    stats = Counter()
    tiny_boxes = 0
    bad_boxes = 0
    tiny_polys = 0
    weak_polys = 0
    class_poly_counts = Counter()
    class_poly_tiny = Counter()
    vis_count = 0

    for imgs_dir, labels_dir, split in sets:
        # Collect images recursively (supports nested class folders)
        img_paths = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            img_paths.extend(glob.glob(os.path.join(
                imgs_dir, "**", ext), recursive=True))
        random.shuffle(img_paths)
        img_paths = img_paths[:max(N_VIS*2, 200)]

        for ip in img_paths:
            # Mirror relative path from images->labels
            rel = os.path.relpath(ip, imgs_dir)
            rel_txt = os.path.splitext(rel)[0] + ".txt"
            lp = os.path.join(labels_dir, rel_txt)
            if not os.path.exists(lp):
                continue
            # keep a simple stem for visualization file naming
            stem = os.path.splitext(os.path.basename(ip))[0]

            try:
                im = Image.open(ip)
                W, H = im.size
            except Exception:
                continue

            anns = []
            with open(lp, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            for ln in lines:
                parts = list(map(float, ln.split()))
                cls = int(parts[0])
                stats[cls] += 1

                if len(parts) == 5:
                    # bbox format: class xc yc w h (normalized)
                    _, xc, yc, w, h = parts
                    x1, y1, x2, y2 = yolo_bbox_norm_to_xyxy(xc, yc, w, h, W, H)
                    if (x2 - x1) <= 1 or (y2 - y1) <= 1:
                        bad_boxes += 1
                    if (x2 - x1) * (y2 - y1) < 4:
                        tiny_boxes += 1
                    anns.append({"cls": cls, "bbox": (x1, y1, x2, y2)})

                elif len(parts) > 5 and len(parts) % 2 == 1:
                    # polygon format: class x1 y1 x2 y2 ... (normalized)
                    coords = parts[1:]
                    poly = norm_to_abs_poly(coords, W, H)
                    area = polygon_area(poly)
                    class_poly_counts[cls] += 1
                    if area < TINY_POLY_AREA_PX:
                        tiny_polys += 1
                        class_poly_tiny[cls] += 1
                    if len(poly) < MIN_POLY_VERTICES:
                        weak_polys += 1
                    anns.append({"cls": cls, "poly": poly})

                else:
                    # Mixed or malformed line
                    pass

            # Save a few visualizations
            if anns and vis_count < N_VIS:
                out_path = os.path.join(OUT_DIR, "yolo", f"{split}_{stem}.jpg")
                draw_vis(im, anns, names, out_path)
                vis_count += 1

    print(f"[YOLO] Class counts:", dict(stats))
    print(
        f"[YOLO] Tiny boxes (<4px area): {tiny_boxes}, Bad boxes (<=1px side): {bad_boxes}")
    print(
        f"[YOLO] Polygons: total={sum(class_poly_counts.values())}, tiny(<{TINY_POLY_AREA_PX}px^2)={tiny_polys}, weak(<{MIN_POLY_VERTICES} verts)={weak_polys}")
    # per-class tiny polygon ratios
    per_class = {int(k): {
        "poly_total": int(class_poly_counts.get(k, 0)),
        "poly_tiny": int(class_poly_tiny.get(k, 0)),
        "poly_tiny_ratio": (class_poly_tiny.get(k, 0) / class_poly_counts.get(k, 1)) if class_poly_counts.get(k, 0) > 0 else 0.0
    } for k in set(list(class_poly_counts.keys()) + list(class_poly_tiny.keys()))}
    # Save summary JSON
    try:
        ensure_dir(OUT_DIR)
        out_json = os.path.join(OUT_DIR, "yolo_summary.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({
                "class_counts": dict(stats),
                "tiny_boxes": tiny_boxes,
                "bad_boxes": bad_boxes,
                "polygons_total": int(sum(class_poly_counts.values())),
                "polygons_tiny": int(tiny_polys),
                "polygons_weak": int(weak_polys),
                "per_class": per_class
            }, f, indent=2)
        print(f"[YOLO] Wrote summary: {out_json}")
    except Exception:
        pass
    return {"format": "YOLO", "class_names": load_yaml_names(data_root), "stats": stats}


def probe_coco(data_root):
    # try to find instances_*.json
    jpaths = find_first([os.path.join(data_root, "**/instances*.json"),
                         os.path.join(data_root, "**/*annotations*.json"),
                         os.path.join(data_root, "**/*coco*.json")])
    if not jpaths:
        return None
    jpath = jpaths[0]
    print(f"[COCO] Found annotation file: {jpath}")
    with open(jpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    images_by_id = {im["id"]: im for im in data.get("images", [])}
    cats = {c["id"]: c.get("name", str(c["id"]))
            for c in data.get("categories", [])}
    print(f"[COCO] Categories: {cats}")

    anns_by_img = defaultdict(list)
    for a in data.get("annotations", []):
        anns_by_img[a["image_id"]].append(a)

    # stats
    counts = Counter([a["category_id"] for a in data.get("annotations", [])])
    print("[COCO] Class counts:", {
          cats.get(k, k): v for k, v in counts.items()})

    # visualize a few
    vis = 0
    for img_id, imrec in images_by_id.items():
        if vis >= N_VIS:
            break
        ip = os.path.join(data_root, imrec.get("file_name", ""))
        if not os.path.exists(ip):
            # try to search
            matches = find_first(
                [os.path.join(data_root, "**", os.path.basename(ip))])
            if not matches:
                continue
            ip = matches[0]

        try:
            im = Image.open(ip)
            W, H = im.size
        except Exception:
            continue

        anns = []
        for a in anns_by_img.get(img_id, []):
            cid = a["category_id"]
            if "bbox" in a:
                x, y, w, h = a["bbox"]
                anns.append({"cls": cid, "bbox": (x, y, x + w, y + h)})
            if "segmentation" in a and a["segmentation"]:
                seg = a["segmentation"]
                if isinstance(seg, list):
                    # polygons
                    # take the first polygon for drawing
                    poly = seg[0]
                    pts = [(max(0, min(W-1, poly[i])), max(0, min(H-1, poly[i+1])))
                           for i in range(0, len(poly), 2)]
                    if len(pts) >= 3:
                        anns.append({"cls": cid, "poly": pts})
                elif isinstance(seg, dict):
                    # RLE
                    if HAS_COCO:
                        rle = coco_mask.frPyObjects(seg, H, W)
                        m = coco_mask.decode(rle)
                        # build polygon outline if you want; for now, skip vis
                    else:
                        pass
        out_path = os.path.join(
            OUT_DIR, "coco", f"{os.path.splitext(os.path.basename(ip))[0]}.jpg")
        draw_vis(im, anns, [cats[k] for k in sorted(cats.keys())], out_path)
        vis += 1

    return {"format": "COCO", "class_names": [cats[k] for k in sorted(cats.keys())], "stats": counts}


def probe_voc(data_root):
    xmls = find_first([os.path.join(data_root, "**/*.xml")])
    if not xmls:
        return None
    try:
        import xml.etree.ElementTree as ET
    except Exception:
        print("[VOC] Cannot import xml parser; skipping.")
        return None

    print(f"[VOC] Found {len(xmls)} XML files (showing up to {N_VIS})")
    counts = Counter()
    vis = 0

    for xp in xmls[:max(50, N_VIS*2)]:
        try:
            tree = ET.parse(xp)
            root = tree.getroot()
            fname = root.findtext("filename")
            ip = os.path.join(os.path.dirname(xp), "..", "JPEGImages", fname)
            if not os.path.exists(ip):
                # fallback: search by filename
                matches = find_first([os.path.join(DATA_ROOT, "**", fname)])
                if not matches:
                    continue
                ip = matches[0]
            im = Image.open(ip)
            W, H = im.size
            anns = []
            for obj in root.findall("object"):
                name = obj.findtext("name")
                counts[name] += 1
                bnd = obj.find("bndbox")
                x1 = float(bnd.findtext("xmin"))
                y1 = float(bnd.findtext("ymin"))
                x2 = float(bnd.findtext("xmax"))
                y2 = float(bnd.findtext("ymax"))
                anns.append({"cls": 0, "bbox": (x1, y1, x2, y2), "name": name})
            out_path = os.path.join(
                OUT_DIR, "voc", f"{os.path.splitext(os.path.basename(ip))[0]}.jpg")
            # pass names as a fake [unique names]
            unique = sorted(set(counts.keys()))
            name_to_idx = {n: i for i, n in enumerate(unique)}
            # remap cls for drawing consistent text
            for a in anns:
                a["cls"] = name_to_idx.get(a["name"], 0)
            draw_vis(im, anns, unique, out_path)
            vis += 1
            if vis >= N_VIS:
                break
        except Exception:
            traceback.print_exc()
            continue

    print("[VOC] Class counts:", dict(counts))
    return {"format": "VOC", "class_names": sorted(set(counts.keys())), "stats": counts}


# =========================
# Main
# =========================
def main():
    ensure_dir(OUT_DIR)
    print(f"Scanning dataset root: {DATA_ROOT}")

    # Try YOLO first (most common for YOLOv8)
    meta = probe_yolo(DATA_ROOT)
    if meta:
        print("\nDetected format: YOLO")
        print("Class names:", meta["class_names"])
        return

    # Then COCO
    meta = probe_coco(DATA_ROOT)
    if meta:
        print("\nDetected format: COCO")
        print("Class names:", meta["class_names"])
        return

    # Then VOC
    meta = probe_voc(DATA_ROOT)
    if meta:
        print("\nDetected format: VOC")
        print("Class names:", meta["class_names"])
        return

    print("❗ Could not detect YOLO/COCO/VOC annotations under the given root. "
          "Please check DATA_ROOT or share your folder tree.")


if __name__ == "__main__":
    main()
