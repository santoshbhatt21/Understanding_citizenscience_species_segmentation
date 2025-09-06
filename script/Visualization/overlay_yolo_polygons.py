"""
Quick viewer to overlay YOLO segmentation polygons on an image.

Usage (PowerShell):
  python script/Visualization/overlay_yolo_polygons.py --image "E:/path/img.jpg" --labels "E:/path/labels/img.txt" --names "E:/path/names.txt" --save "E:/path/overlay.jpg"

Notes:
- Label format: each line = class_id x1 y1 x2 y2 ... (normalized [0..1])
- names.txt is optional; if provided, one class name per line in index order.
"""

import argparse
import os
from typing import List, Tuple, Optional

import cv2
import numpy as np


def read_names(names_path: Optional[str]) -> Optional[List[str]]:
    if not names_path:
        return None
    if not os.path.isfile(names_path):
        return None
    with open(names_path, "r", encoding="utf-8") as f:
        names = [ln.strip() for ln in f.readlines() if ln.strip()]
    return names or None


def parse_yolo_seg_txt(txt_path: str) -> List[Tuple[int, np.ndarray]]:
    """Returns list of (class_id, polygon_norm) where polygon_norm is Nx2 float array in [0,1]."""
    objs: List[Tuple[int, np.ndarray]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 7:
                # class + at least 3 points (6 numbers)
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            coords = parts[1:]
            if len(coords) % 2 == 1:
                # drop the last if odd count
                coords = coords[:-1]
            pts = np.array(coords, dtype=float).reshape(-1, 2)
            if pts.shape[0] < 3:
                continue
            # Clamp to [0,1]
            pts = np.clip(pts, 0.0, 1.0)
            objs.append((cls, pts))
    return objs


def color_for_class(cid: int) -> Tuple[int, int, int]:
    # Distinct-ish colors via HSV -> BGR
    h = (cid * 37) % 180  # 0..179
    s = 200
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def overlay_polygons(
    img_bgr: np.ndarray,
    objects: List[Tuple[int, np.ndarray]],
    class_names: Optional[List[str]] = None,
    alpha: float = 0.4,
) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    overlay = img_bgr.copy()

    for cls, poly_norm in objects:
        # Normalize -> pixel coords
        poly_px = poly_norm.copy()
        poly_px[:, 0] *= w
        poly_px[:, 1] *= h
        poly_px = np.round(poly_px).astype(int)
        if poly_px.shape[0] < 3:
            continue
        pts = poly_px.reshape(-1, 1, 2)

        color = color_for_class(cls)
        # Filled polygon on overlay
        cv2.fillPoly(overlay, [pts], color=color)
        # Outline on base image
        cv2.polylines(img_bgr, [pts], isClosed=True,
                      color=color, thickness=2, lineType=cv2.LINE_AA)

        # Class label near first point
        label = class_names[cls] if class_names and 0 <= cls < len(
            class_names) else f"{cls}"
        x0, y0 = int(pts[0, 0, 0]), int(pts[0, 0, 1])
        cv2.putText(
            img_bgr,
            label,
            (x0 + 3, max(12, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    # Blend overlay onto image
    blended = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
    return blended


def main():
    ap = argparse.ArgumentParser(
        description="Overlay YOLO segmentation polygons on an image")
    ap.add_argument("--image", required=True, help="Path to the image file")
    ap.add_argument("--labels", required=True,
                    help="Path to the YOLO .txt labels for this image")
    ap.add_argument("--names", default=None,
                    help="Optional names.txt (one class name per line)")
    ap.add_argument("--alpha", type=float, default=0.4,
                    help="Fill transparency [0..1], default 0.4")
    ap.add_argument("--save", default=None,
                    help="Optional output path to save the overlay image")
    ap.add_argument("--no_show", action="store_true",
                    help="Do not display window; useful for batch save")
    args = ap.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.isfile(args.labels):
        raise FileNotFoundError(f"Labels not found: {args.labels}")

    class_names = read_names(args.names)
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    objects = parse_yolo_seg_txt(args.labels)
    if not objects:
        print("No polygons found in labels; showing original image.")
        result = img
    else:
        result = overlay_polygons(
            img, objects, class_names=class_names, alpha=args.alpha)

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        cv2.imwrite(args.save, result)
        print(f"Saved: {args.save}")

    if not args.no_show:
        cv2.imshow("Overlay", result)
        print("Press any key in the image window to close…")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
