'''import os, glob, re
from PIL import Image, UnidentifiedImageError

# ── CONFIG ───────────────────────────────────────────
IMAGES_DIR = r"E:/Santosh_master_thesis/LT_species_organ_10_species/Abies alba leaves"   # folder of the 2026 images
MASKS_DIR  = r"E:/Santosh_master_thesis/LT_species_organ_10_species/Abies alba leaves_mask"     # where 1223 masks live
LABELS_DIR = r"E:/Santosh_master_thesis/LT_species_organ_10_species/Abies alba leaves_labels"   # where 1223 yolo-seg txt live

IMG_EXTS   = (".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp")
MASK_EXTS  = (".png",".jpg",".jpeg")
# filename patterns you used for masks:
MASK_PREFIX = "mask_"      # e.g., mask_IMG_001.png  (set "" if not used)
MASK_SUFFIX = "_mask"      # e.g., IMG_001_mask.png (set "" if not used)
BACKGROUND_VALUE = 255     # expected "no-object" value in your pipeline
MIN_POLY_POINTS = 3        # exporter requires ≥3 vertices per polygon
# ────────────────────────────────────────────────────

def stem(p): 
    n = os.path.basename(p)
    return os.path.splitext(n)[0]

def candidate_mask_paths(st):
    cands = []
    # 1) same stem
    for ext in MASK_EXTS:
        cands.append(os.path.join(MASKS_DIR, st + ext))
    # 2) prefix
    for ext in MASK_EXTS:
        cands.append(os.path.join(MASKS_DIR, MASK_PREFIX + st + ext))
    # 3) suffix
    for ext in MASK_EXTS:
        cands.append(os.path.join(MASKS_DIR, st + MASK_SUFFIX + ext))
    return cands

def has_mask_file(st):
    for p in candidate_mask_paths(st):
        if os.path.exists(p):
            return p
    return None

def mask_has_foreground(p, bg=BACKGROUND_VALUE):
    try:
        im = Image.open(p).convert("L")
        # quick check: any pixel != bg?
        # Use a tiny sample to avoid full load if huge:
        arr = im.resize((max(1,im.width//8), max(1,im.height//8)))
        vals = arr.getdata()
        return any(v != bg for v in vals)
    except Exception:
        return False

def label_ok(p):
    if not os.path.exists(p): 
        return False, "absent"
    try:
        with open(p, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            return False, "empty-txt"
        for ln in lines:
            parts = ln.split()
            if len(parts) < 1 + 2*MIN_POLY_POINTS:
                return False, "degenerate-poly"
            # basic numeric check
            try:
                cid = int(parts[0])
                coords = list(map(float, parts[1:]))
                if any((c<0 or c>1) for c in coords):
                    return False, "unnormalized"
            except Exception:
                return False, "parse-error"
        return True, "ok"
    except Exception:
        return False, "read-error"

all_imgs = sorted([p for p in glob.glob(os.path.join(IMAGES_DIR,"**","*"), recursive=True)
                   if os.path.splitext(p)[1].lower() in IMG_EXTS])

missing = []
reasons = {}
for ip in all_imgs:
    st = stem(ip)
    mp = has_mask_file(st)
    if not mp:
        reasons.setdefault("no-mask-file", []).append(ip); continue
    if not mask_has_foreground(mp):
        reasons.setdefault("mask-no-foreground", []).append(ip); continue
    # label with same basename expected
    lp = os.path.join(LABELS_DIR, st + ".txt")
    ok, why = label_ok(lp)
    if not ok:
        reasons.setdefault(f"label-{why}", []).append(ip); continue

# Summary
total = len(all_imgs)
missing_ct = sum(len(v) for v in reasons.values())
print(f"Images: {total}, Problematic: {missing_ct}")
for k,v in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
    print(f"{k:>22}: {len(v)}")
# (Optional) write lists to files to spot-check a few examples per reason
for k,v in reasons.items():
    out = os.path.join(os.path.dirname(IMAGES_DIR), f"debug_{k}.txt")
    with open(out,"w",encoding="utf-8") as f:
        f.write("\n".join(v))
print("Wrote debug_*.txt next to your images folder.")'''

import os, glob, cv2
import numpy as np
from pathlib import Path

IMAGES_DIR = r"E:/Santosh_master_thesis/LT_species_organ_10_species/Abies alba leaves"
MASKS_DIR  = r"E:/Santosh_master_thesis/LT_species_organ_10_species/Abies alba leaves"
LABELS_DIR = r"E:/Santosh_master_thesis/LT_species_organ_10_species/Abies alba leaves"
CLASS_ID   = 0  # <-- put the correct final class id for "Abies alba leaves"

def ensure_dir(d): Path(d).mkdir(parents=True, exist_ok=True)

def to_yolo_poly(contour, w, h):
    eps = 0.002 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps, True)
    xs = approx[:,0,0] / w
    ys = approx[:,0,1] / h
    pts = np.stack([xs, ys], axis=1).reshape(-1)
    return pts.tolist()

def convert_one(stem):
    imgp = None
    for ext in (".jpg",".jpeg",".png",".tif",".tiff",".bmp",".webp"):
        p = Path(IMAGES_DIR)/f"{stem}{ext}"
        if p.exists(): imgp = str(p); break
    if imgp is None: return False, "no-image"

    maskp = Path(MASKS_DIR)/f"{stem}.png"
    if not maskp.exists(): return False, "no-mask"

    im = cv2.imread(imgp, cv2.IMREAD_COLOR)
    m  = cv2.imread(str(maskp), cv2.IMREAD_GRAYSCALE)
    if im is None or m is None: return False, "io-error"
    h, w = m.shape[:2]

    # assume foreground!=255 background; invert if needed
    # make binary
    _, mb = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cnts,_ = cv2.findContours(mb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts: 
        return False, "empty"

    # pick largest or write multiple lines if multiple instances per image
    lines = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.001 * w*h:  # drop tiny specks; tune as you like
            continue
        pts = to_yolo_poly(c, w, h)
        if len(pts) < 6: 
            continue
        line = " ".join([str(CLASS_ID)] + [f"{x:.6f}" for x in pts])
        lines.append(line)

    if not lines:
        return False, "degenerate"

    ensure_dir(LABELS_DIR)
    with open(Path(LABELS_DIR)/f"{stem}.txt","w",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    return True, "ok"

def main():
    stems = [Path(p).stem for p in glob.glob(os.path.join(MASKS_DIR,"*.png"))]
    ok, bad = 0, 0
    for st in stems:
        success, why = convert_one(st)
        if success: ok+=1
        else: bad+=1
    print(f"labels ok: {ok}, failed: {bad}")

if __name__ == "__main__":
    main()

