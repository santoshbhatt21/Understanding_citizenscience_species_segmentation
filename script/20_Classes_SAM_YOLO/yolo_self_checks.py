import os, glob, math

DATASET = "E:/Santosh_master_thesis/LT_species_organ_10_species"  # folder with images/ and labels/
SPLITS = ["train", "val"]
NC = 20  # number of classes in your data.yaml

def pairs(seq): 
    return list(zip(seq[0::2], seq[1::2]))

def poly_area_norm(pts):
    # Shoelace area in normalized space
    if len(pts) < 3: return 0.0
    s = 0.0
    for i in range(len(pts)):
        x1,y1 = pts[i]
        x2,y2 = pts[(i+1)%len(pts)]
        s += x1*y2 - x2*y1
    return abs(s) / 2.0

issues = []
for split in SPLITS:
    img_dir = os.path.join(DATASET, "images", split)
    lbl_dir = os.path.join(DATASET, "labels", split)
    for img in glob.glob(os.path.join(img_dir, "**", "*.*"), recursive=True):
        if os.path.splitext(img)[1].lower() not in [".jpg",".jpeg",".png",".bmp",".tif",".tiff"]:
            continue
        base = os.path.splitext(os.path.relpath(img, img_dir))[0]
        lbl = os.path.join(lbl_dir, base + ".txt")
        if not os.path.isfile(lbl):
            issues.append(("missing_label", img))
            continue
        with open(lbl, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                try:
                    parts = line.strip().split()
                    if not parts: 
                        issues.append(("empty_line", lbl, ln)); continue
                    cls = int(parts[0])
                    if cls < 0 or cls >= NC:
                        issues.append(("class_oob", lbl, ln, cls))
                    nums = list(map(float, parts[1:]))
                    if len(nums) % 2 != 0 or len(nums) < 6:
                        issues.append(("bad_poly_len", lbl, ln, len(nums))); continue
                    pts = pairs(nums)
                    bad_xy = [(x,y) for x,y in pts if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0)]
                    if bad_xy:
                        issues.append(("out_of_range_xy", lbl, ln, bad_xy[:3]))
                    area = poly_area_norm(pts)
                    if area < 1e-5:
                        issues.append(("tiny_or_degenerate", lbl, ln, area))
                except Exception as e:
                    issues.append(("parse_error", lbl, ln, str(e)))

# Summarize
from collections import Counter
cnt = Counter(t[0] for t in issues)
print("Issue counts:", cnt)
for t in issues[:200]:
    print(t)
