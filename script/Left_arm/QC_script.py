import os
import cv2
import numpy as np
from shapely.geometry import Polygon
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
OVERLAY_DIR = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM_overlays"
BAD_OUT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/QC_bad_overlays"
CSV_OUT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/QC_overlay_report.csv"

os.makedirs(BAD_OUT, exist_ok=True)

# ============================================================
# HELPERS
# ============================================================
def touches_border(poly, w, h):
    xs = poly[:,0]; ys = poly[:,1]
    if np.min(xs) <= 1 or np.max(xs) >= w-2:
        return True
    if np.min(ys) <= 1 or np.max(ys) >= h-2:
        return True
    return False

def polygon_smoothness(poly):
    diffs = np.diff(poly, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    return np.mean(norms), np.std(norms)

def fragmentation(mask):
    num_labels, labels = cv2.connectedComponents(mask)
    return num_labels - 1

def simplify_polygon(poly):
    approx = cv2.approxPolyDP(poly.reshape(-1,1,2).astype(np.int32), epsilon=1, closed=True)
    return approx.reshape(-1,2)

# ============================================================
# MAIN QC
# ============================================================
records = []

for fname in os.listdir(OVERLAY_DIR):
    if not fname.lower().endswith((".jpg",".png",".jpeg")):
        continue

    fpath = os.path.join(OVERLAY_DIR, fname)
    img = cv2.imread(fpath)
    if img is None:
        continue

    h, w = img.shape[:2]

    # green outline mask extraction (your overlays use green polygon)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([40, 40, 40])
    upper = np.array([80, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # find contour
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        records.append([fname, "NO_POLYGON", None, None, None, None, None])
        cv2.imwrite(os.path.join(BAD_OUT, fname), img)
        continue

    cnt = max(contours, key=cv2.contourArea)

    if cv2.contourArea(cnt) < 50:
        records.append([fname, "TOO_SMALL", cv2.contourArea(cnt), None, None, None, None])
        cv2.imwrite(os.path.join(BAD_OUT, fname), img)
        continue

    poly = cnt.reshape(-1,2)

    # polygon validity
    shp = Polygon(poly)
    valid = shp.is_valid

    # border check
    border = touches_border(poly, w, h)

    # smoothness check
    mean_step, std_step = polygon_smoothness(poly)

    # fragmentation check
    frag = fragmentation(mask)

    status = "GOOD"
    if border: status = "BORDER_TOUCH"
    if not valid: status = "INVALID_POLY"
    if frag > 1: status = "MULTIPLE_MASKS"

    # save bad examples
    if status != "GOOD":
        cv2.imwrite(os.path.join(BAD_OUT, fname), img)

    records.append([
        fname,
        status,
        cv2.contourArea(cnt),
        len(poly),
        mean_step,
        std_step,
        frag
    ])

# save report
df = pd.DataFrame(records, columns=[
    "file", "status", "area", "num_points", "mean_step", "std_step", "fragments"
])
df.to_csv(CSV_OUT, index=False)

print("\n====================================")
print("QC Completed")
print(f"Report saved to: {CSV_OUT}")
print(f"Bad overlays saved to: {BAD_OUT}")
print("====================================\n")
