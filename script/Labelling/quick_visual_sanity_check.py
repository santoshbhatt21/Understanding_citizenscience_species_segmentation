import os
import random
from glob import glob

import cv2
import numpy as np

# Roots: images remain under DATA_YOLOv8; labels use the cleaned folder
IMAGES_ROOT = r"E:/Santosh_master_thesis/DATA_YOLOv8/images"
LABELS_ROOT = r"E:/Santosh_master_thesis/DATA_YOLOv8_clean/labels"
OUT_DIR = r"E:/Santosh_master_thesis"


def overlay_one(img_path: str, lbl_path: str, save_path: str):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Skip unreadable image: {img_path}")
        return False
    h, w = img.shape[:2]
    if os.path.exists(lbl_path):
        with open(lbl_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                except Exception:
                    continue
                coords = list(map(float, parts[1:]))
                if len(coords) < 6 or len(coords) % 2 != 0:
                    continue
                xy = np.array(coords, dtype=np.float32).reshape(-1, 2)
                pts = (xy * [w, h]).astype(int)
                cv2.polylines(img, [pts], isClosed=True,
                              color=(0, 0, 255), thickness=3)
                cv2.putText(img, str(cls), tuple(
                    pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "NO LABEL", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)
    return True


def main():
    # collect candidates from val first, then train
    img_patterns = [
        os.path.join(IMAGES_ROOT, 'val', '**', '*.*'),
        os.path.join(IMAGES_ROOT, 'train', '**', '*.*'),
    ]
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    images = []
    for pat in img_patterns:
        images.extend([p for p in glob(pat, recursive=True)
                      if os.path.splitext(p)[1].lower() in img_exts])

    random.shuffle(images)
    sample = images[:10] if len(images) >= 10 else images
    if not sample:
        print("No images found under", IMAGES_ROOT)
        return

    saved = 0
    for img_path in sample:
        # derive label path by mirroring relative path under labels root
        rel = os.path.relpath(img_path, start=os.path.join(IMAGES_ROOT))
        rel_no_ext = os.path.splitext(rel)[0] + '.txt'
        lbl_path = os.path.join(LABELS_ROOT, rel_no_ext)
        out_name = os.path.basename(
            os.path.splitext(img_path)[0]) + '_overlay.jpg'
        save_path = os.path.join(OUT_DIR, 'debug_overlays', out_name)
        if overlay_one(img_path, lbl_path, save_path):
            saved += 1

    print(f"Saved {saved} overlays to {os.path.join(OUT_DIR, 'debug_overlays')}")


if __name__ == '__main__':
    main()
