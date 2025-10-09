#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grad-CAM → SAM → YOLO (Leaves-only, single-head classifier)
with mask merging/cleaning + separate labels root

Fixes:
- CAM upsampled to original image resolution before overlay and mask thresholding
  (avoids shape broadcast error and coordinate mismatch with SAM points).

Run:
    python cam_sam_yolo_leaves_singlehead_cleanmerge.py

Outputs (per class):
    OUT_CAMS/<class>/<stem>_cam.png
    OUT_MASKS/<class>/<stem>_mask.png
    OUT_LABELS/<class>/<stem>.txt   (YOLO polygon labels matching original image stem)
"""

# ===========================
# CONFIG — EDIT THESE
# ===========================
CHECKPOINT = r"E:/Santosh_master_thesis/Checkpoints_Leaves_OneCycle_F1_Temp_bestCM/best_by_loss_ep25_0.624.pth"
BASE_DIR   = r"E:/Santosh_master_thesis/classified_Leaves"  # 10 class folders

OUT_CAMS    = r"E:/Santosh_master_thesis/output_cams"
OUT_MASKS   = r"E:/Santosh_master_thesis/output_masks"
OUT_LABELS  = r"E:/Santosh_master_thesis/output_labels"

# SAM
SAM_CHECKPOINT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"   # "vit_h" | "vit_l" | "vit_b"

# CAM / image
IMG_SIZE      = 640
DEVICE        = "cuda"  # or "cpu"
CAM_METHOD    = "gradcam"   # "hirescam" | "gradcam++" | "xgradcam" | "gradcam"
CAM_PCT       = 85          # percentile threshold (0–100) on upsampled CAM
MIN_AREA_FRAC = 0.002       # drop CAM blobs smaller than this fraction of image area
DILATE_PX     = 14
ERODE_PX      = 6

# SAM→merge
IOU_THRESH_CAM = 0.30       # keep SAM masks with IoU >= this vs CAM mask
TOPK_SAM       = 3          # union at most top-K SAM masks by IoU
MORPH_CLOSE    = 11         # 0 disables
MORPH_OPEN     = 5          # 0 disables
FILL_HOLES     = True
REMOVE_ISLANDS_MIN_FRAC = 0.002  # remove tiny components after merge

# Polygons
MAX_POLYGONS_PER_IMAGE = 3
MIN_POLY_POINTS = 6
SIMPLIFY_EPS_FRAC = 0.002
FORCE_SINGLE_POLYGON = False  # if True: single convex hull polygon

# Debug/limit
LIMIT_IMAGES_PER_CLASS = None
VERBOSE = True

# ===========================
# IMPLEMENTATION
# ===========================
import os, glob
from pathlib import Path
from typing import List
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import cv2

# Grad-CAM (robust import)
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, HiResCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
try:
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    class ClassifierOutputTarget:  # fallback
        def __init__(self, category: int):
            self.category = category
        def __call__(self, model_output):
            return model_output[:, self.category]

# SAM
from segment_anything import sam_model_registry, SamPredictor


def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def imread_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def preprocess_pil(img: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.12)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tfm(img)

def remove_small_components(mask01: np.ndarray, min_area_frac: float) -> np.ndarray:
    H, W = mask01.shape
    min_area = int(min_area_frac * H * W)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, connectivity=8)
    if num_labels <= 1:
        return mask01
    keep = np.zeros_like(mask01)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= max(min_area, 1):
            keep[labels == i] = 1
    return keep

def binarize_cam(cam: np.ndarray, pct: float, min_area_frac: float) -> np.ndarray:
    thr = np.percentile(cam, pct)
    m = (cam >= thr).astype(np.uint8)
    m = remove_small_components(m, min_area_frac)
    if DILATE_PX > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATE_PX, DILATE_PX)))
    if ERODE_PX > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ERODE_PX, ERODE_PX)))
    return m

def iou(a: np.ndarray, b: np.ndarray) -> float:
    A = a.astype(bool); B = b.astype(bool)
    inter = (A & B).sum()
    uni   = (A | B).sum()
    return float(inter) / float(uni + 1e-6)

def fill_holes(mask01: np.ndarray) -> np.ndarray:
    H, W = mask01.shape
    flood = mask01.copy().astype(np.uint8)
    buf = np.zeros((H+2, W+2), np.uint8)
    cv2.floodFill(flood, buf, (0,0), 1)
    inv = 1 - flood
    return np.logical_or(mask01, inv).astype(np.uint8)

def merge_sam_masks(cam_mask: np.ndarray, sam_masks: List[np.ndarray],
                    iou_thresh: float, topk: int) -> np.ndarray:
    scored = [(iou(cam_mask, m), m) for m in sam_masks]
    scored.sort(key=lambda x: x[0], reverse=True)
    kept = [m for s,m in scored if s >= iou_thresh][:topk]
    if not kept:
        merged = (scored[0][1] > 0).astype(np.uint8) if scored else np.zeros_like(cam_mask)
    else:
        merged = np.zeros_like(cam_mask)
        for m in kept:
            merged = np.logical_or(merged, m > 0).astype(np.uint8)

    if MORPH_CLOSE and MORPH_CLOSE > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_CLOSE, MORPH_CLOSE))
        merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k)
    if MORPH_OPEN and MORPH_OPEN > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_OPEN, MORPH_OPEN))
        merged = cv2.morphologyEx(merged, cv2.MORPH_OPEN, k)
    if FILL_HOLES:
        merged = fill_holes(merged)
    if REMOVE_ISLANDS_MIN_FRAC and REMOVE_ISLANDS_MIN_FRAC > 0:
        merged = remove_small_components(merged, REMOVE_ISLANDS_MIN_FRAC)
    return merged

def contours_to_yolo_polys(mask: np.ndarray, max_polys=3, force_single=False) -> List[List[float]]:
    H, W = mask.shape
    cnts,_ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if force_single and len(cnts) > 0:
        all_pts = np.vstack(cnts).reshape(-1,2)
        hull = cv2.convexHull(all_pts)
        cnts = [hull]

    areas = [cv2.contourArea(c) for c in cnts]
    order = np.argsort(areas)[::-1]
    polys = []
    eps = SIMPLIFY_EPS_FRAC * max(H, W)
    for k in order[:max_polys]:
        c = cnts[k]
        if len(c) < MIN_POLY_POINTS or cv2.contourArea(c) <= 1:
            continue
        c = cv2.approxPolyDP(c, epsilon=eps, closed=True)
        pts = c.reshape(-1,2).astype(np.float32)
        pts[:,0] = np.clip(pts[:,0] / W, 0, 1)
        pts[:,1] = np.clip(pts[:,1] / H, 0, 1)
        flat = pts.flatten().tolist()
        if len(flat) >= 6:
            polys.append(flat)
    return polys

def load_single_head_model(ckpt_path, num_classes):
    m = models.efficientnet_v2_s(weights=None)
    in_feats = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_feats, num_classes))
    sd = torch.load(ckpt_path, map_location="cpu")
    sd = sd.get("state_dict") or sd.get("model") or sd
    if isinstance(sd, dict):
        sd = {k.replace("module.", ""): v for k,v in sd.items()}
    m.load_state_dict(sd, strict=False)
    return m

def get_target_layers(model):
    if hasattr(model, "features"):
        return [model.features[-1]]
    raise ValueError("Cannot determine target layer for Grad-CAM.")

def main():
    base = Path(BASE_DIR)
    classes = [d for d in sorted(os.listdir(base)) if (base/d).is_dir()]
    assert len(classes) > 1, "No class folders found under BASE_DIR."

    for r in (OUT_CAMS, OUT_MASKS, OUT_LABELS):
        Path(r).mkdir(parents=True, exist_ok=True)

    model = load_single_head_model(CHECKPOINT, num_classes=len(classes)).to(DEVICE).eval()

    target_layers = get_target_layers(model)
    method = CAM_METHOD.lower()
    if method == "hirescam":
        CAMcls = HiResCAM
    elif method == "gradcam++":
        CAMcls = GradCAMPlusPlus
    elif method == "xgradcam":
        CAMcls = XGradCAM
    else:
        CAMcls = GradCAM
    cam = CAMcls(model=model, target_layers=target_layers)

    if not os.path.isfile(SAM_CHECKPOINT):
        raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CHECKPOINT}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT).to(device=DEVICE)
    predictor = SamPredictor(sam)

    print(f"Found {len(classes)} class folders.")
    for cls_name in classes:
        cls_dir = base / cls_name
        cams_cls   = Path(OUT_CAMS)   / cls_name
        masks_cls  = Path(OUT_MASKS)  / cls_name
        labels_cls = Path(OUT_LABELS) / cls_name
        for d in (cams_cls, masks_cls, labels_cls): ensure_dir(d)

        class_id = classes.index(cls_name)

        image_paths = []
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp"):
            image_paths.extend(glob.glob(str(cls_dir / ext)))
        image_paths.sort()
        if LIMIT_IMAGES_PER_CLASS is not None:
            image_paths = image_paths[:LIMIT_IMAGES_PER_CLASS]

        for p in tqdm(image_paths, desc=f"{cls_name}", unit="img"):
            img = imread_rgb(p)
            W, H = img.size
            x = preprocess_pil(img).unsqueeze(0).to(DEVICE)

            grayscale_cam = cam(
                input_tensor=x,
                targets=[ClassifierOutputTarget(class_id)],
                eigen_smooth=False,
                aug_smooth=False
            )[0]
            gmin, gmax = float(grayscale_cam.min()), float(grayscale_cam.max())
            cam01 = (grayscale_cam - gmin) / (gmax - gmin + 1e-6)  # (IMG_SIZE, IMG_SIZE)

            # Upsample CAM to original image resolution BEFORE thresholding & overlay
            cam_full = cv2.resize(cam01, (W, H), interpolation=cv2.INTER_LINEAR)

            # Overlay (original size)
            rgb_np = np.array(img).astype(np.float32) / 255.0
            cam_rgb = show_cam_on_image(rgb_np, cam_full, use_rgb=True)
            cv2.imwrite(str(cams_cls / f"{Path(p).stem}_cam.png"), cv2.cvtColor(cam_rgb, cv2.COLOR_RGB2BGR))

            # Binary mask from full-resolution CAM
            cam_mask = binarize_cam(cam_full, pct=CAM_PCT, min_area_frac=MIN_AREA_FRAC)

            # SAM points from cam_mask (same resolution as original now)
            ys, xs = np.where(cam_mask == 1)
            predictor.set_image(np.array(img))
            if len(xs) == 0:
                pos = np.array([[W//2, H//2]])
                labels = np.ones((1,), dtype=np.int32)
            else:
                sel = np.random.choice(len(xs), size=min(64, len(xs)), replace=False)
                pos = np.stack([xs[sel], ys[sel]], axis=1)
                labels = np.ones((len(sel),), dtype=np.int32)

            sam_masks, _, _ = predictor.predict(point_coords=pos, point_labels=labels, box=None, multimask_output=True)

            merged = merge_sam_masks(cam_mask, list(sam_masks), IOU_THRESH_CAM, TOPK_SAM)
            cv2.imwrite(str(masks_cls / f"{Path(p).stem}_mask.png"), (merged * 255).astype(np.uint8))

            polys = contours_to_yolo_polys(merged, max_polys=MAX_POLYGONS_PER_IMAGE,
                                           force_single=FORCE_SINGLE_POLYGON)
            if not polys:
                continue

            with open(labels_cls / f"{Path(p).stem}.txt", "w", encoding="utf-8") as f:
                for poly in polys:
                    f.write(str(class_id) + " " + " ".join(f"{v:.6f}" for v in poly) + "\n")

    print("Done.\nCAMS :", OUT_CAMS, "\nMASKS:", OUT_MASKS, "\nLABELS:", OUT_LABELS)


if __name__ == "__main__":
    main()