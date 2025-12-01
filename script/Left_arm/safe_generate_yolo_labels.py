import os
import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

# ============================================================
# USER PATHS
# ============================================================
IMG_ROOT   = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
MASK_ROOT  = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data_masks_clean_safe"
LABEL_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labels_10class_clean_safe_FIXED"
OVERLAY_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM_overlays"

SAVE_OVERLAY = True
os.makedirs(LABEL_ROOT, exist_ok=True)
os.makedirs(OVERLAY_ROOT, exist_ok=True)


# ============================================================
# LOGGING UTIL
# ============================================================
def log(msg):
    safe_msg = msg.encode("ascii","replace").decode()
    print(safe_msg)


# ============================================================
# MASK VALIDATION
# ============================================================
def mask_invalid(mask):
    """Reject empty, full, tiny, touching-border masks."""
    u = np.unique(mask)
    if len(u) == 1:  
        return True
    if np.sum(mask == 255) < 60:
        return True
    return False


def touches_border(cnt, w, h):
    """Reject contours that touch the image edges."""
    xs = cnt[:,0,0]
    ys = cnt[:,0,1]
    if np.min(xs) <= 1 or np.max(xs) >= w-2:
        return True
    if np.min(ys) <= 1 or np.max(ys) >= h-2:
        return True
    return False


# ============================================================
# POLYGON EXTRACTION + CLEANING
# ============================================================
def simplify_polygon(contour, epsilon=2.0):
    """Ramer–Douglas–Peucker simplification."""
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx.reshape(-1,2)


def find_best_polygon(mask, w, h):
    """Find the *best* SAM mask contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        return None

    # filter invalid contours
    valid = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        if touches_border(cnt, w, h):
            continue
        valid.append(cnt)

    if len(valid) == 0:
        return None

    # choose the largest clean contour
    best = max(valid, key=cv2.contourArea)

    # simplify
    poly = simplify_polygon(best)

    # ensure polygon is >=10 points
    if len(poly) < 10:
        return None

    # fix self-intersection
    try:
        shp = Polygon(poly)
        if not shp.is_valid:
            shp = shp.buffer(0)
        x, y = shp.exterior.coords.xy
        poly = np.vstack([x, y]).T
    except:
        return None

    return poly


# ============================================================
# YOLO WRITE
# ============================================================
def save_yolo_polygon(label_path, class_id, poly, w, h):
    with open(label_path, "w") as f:
        f.write(f"{class_id} ")
        for x, y in poly:
            f.write(f"{x/w:.6f} {y/h:.6f} ")
    return


# ============================================================
# OPTIONAL OVERLAY
# ============================================================
def save_overlay(img, poly, out_path):
    over = img.copy()
    pts = poly.reshape(-1,1,2).astype(np.int32)
    cv2.polylines(over, [pts], isClosed=True, color=(0,255,0), thickness=2)
    cv2.imwrite(out_path, over)


# ============================================================
# MAIN
# ============================================================
def generate_labels():

    species_folders = sorted([d for d in os.listdir(IMG_ROOT)
                              if os.path.isdir(os.path.join(IMG_ROOT, d))])
    species_to_class = {sp:i for i, sp in enumerate(species_folders)}

    log("=== CLASS MAPPING ===")
    for sp,cid in species_to_class.items():
        log(f"{cid} → {sp}")
    log("======================\n")

    for species in species_folders:
        log(f"\n--- Processing species: {species} ---")

        class_id = species_to_class[species]
        img_dir  = os.path.join(IMG_ROOT, species)
        mask_dir = os.path.join(MASK_ROOT, species + "_mask")
        out_dir  = os.path.join(LABEL_ROOT, species)

        if not os.path.isdir(mask_dir):
            log(f"[SKIP] Missing mask_dir for {species}")
            continue

        os.makedirs(out_dir, exist_ok=True)

        image_files = [f for f in os.listdir(img_dir)
                       if f.lower().endswith((".jpg",".jpeg",".png"))]

        for img_name in image_files:
            stem = os.path.splitext(img_name)[0]
            img_path  = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, f"mask_{stem}.png")

            if not os.path.exists(mask_path):
                log(f"[NO MASK] {mask_path}")
                continue

            img = cv2.imread(img_path)
            h,w = img.shape[:2]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask_invalid(mask):
                log(f"[INVALID MASK] {mask_path}")
                continue

            poly = find_best_polygon(mask, w, h)
            if poly is None:
                log(f"[INVALID POLY] {mask_path}")
                continue

            # Save YOLO label
            label_path = os.path.join(out_dir, f"{stem}.txt")
            save_yolo_polygon(label_path, class_id, poly, w, h)

            # overlay
            if SAVE_OVERLAY:
                overlay_path = os.path.join(OVERLAY_ROOT, f"{species}_{stem}.jpg")
                save_overlay(img, poly, overlay_path)

            log(f"[OK] {label_path}")


if __name__ == "__main__":
    generate_labels()
    log("\n✓ DONE — All YOLO-Seg labels fixed & cleaned.")
