# -*- coding: utf-8 -*-
"""
CAM -> SAM mask & YOLO polygon exporter (class-aware, robust)
- Loads "best_model*.pth" automatically and adapts to its head size
- Grad-CAM with optional hflip TTA (averaged CAMs)
- Class-aware CAM thresholds and min-area (leaves vs trunks)
- Confidence gating: softmax(target) >= threshold AND top1-top2 >= gap
- Stable SAM prompts: centroid + farthest boundary points
- Saves:
    <base_dir>/<class>_mask/.../mask_<img>.png
    <base_dir>/<class>_labels/.../<img>.txt
"""

import os
import re
import json
import logging
from collections import OrderedDict
from typing import Optional, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from segment_anything import sam_model_registry, SamPredictor

# =========================
# Logging
# =========================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cam_sam_export")

# =========================
# Paths (EDIT THESE)
# =========================
base_dir = r"E:/Santosh_master_thesis/LT_species_organ_10_species"
checkpoint_dir = r"E:/Santosh_master_thesis/Checkpoints_species_organ_weighted_random_sampler_focal_loss"
sam_checkpoint = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"

# =========================
# Processing knobs
# =========================
# CAM
CAM_INPUT_SIZE = 512                 # try 384 if VRAM is tight
BLUR_SIGMA = 2.0
USE_CAM_TTA = True                   # avg with hflip CAMs for denoising
USE_AMP = True                       # CUDA AMP for speed

# Class-aware policies (leaves vs trunks)
LEAF_Q = 0.85                        # percentile threshold for leaves
TRUNK_Q = 0.90                       # stricter for trunks
LEAF_MIN_AREA = 0.001                # 0.1% of image
TRUNK_MIN_AREA = 0.001               # 0.1% for trunks

CONF_THRESH_LEAF = 0.90              # min softmax prob for target class
CONF_THRESH_TRUNK = 0.95
TOP2_GAP_MIN = 0.20                  # top1 - top2 softmax gap

# SAM
PROMPT_POINTS = 3                    # centroid + up to 2 farthest boundary pts
# keep False for speed; True is slower but sometimes crisper
SAM_TWO_STAGE = False

# Batching / IO
BATCH_SIZE = 24
LOG_EVERY_N = 25
Background_value = 255               # background fill in mask
image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff',
              '.JPG', '.PNG', '.JPEG')

# Optional threshold profiles to quickly trade recall vs precision.
# Set THRESHOLD_PROFILE to one of: "balanced" (use above), "recall", "precision".
THRESHOLD_PROFILE = "recall"


def _apply_threshold_profile():
    global LEAF_Q, TRUNK_Q, LEAF_MIN_AREA, TRUNK_MIN_AREA
    global CONF_THRESH_LEAF, CONF_THRESH_TRUNK, TOP2_GAP_MIN
    global SAM_TWO_STAGE, BLUR_SIGMA
    if THRESHOLD_PROFILE == "recall":
        # Looser thresholds to keep more samples
        LEAF_Q = min(0.90, max(0.70, LEAF_Q - 0.03))
        TRUNK_Q = min(0.90, max(0.70, TRUNK_Q - 0.03))
        LEAF_MIN_AREA = max(0.0008, LEAF_MIN_AREA * 0.75)
        TRUNK_MIN_AREA = max(0.0008, TRUNK_MIN_AREA * 0.75)
        CONF_THRESH_LEAF = max(0.80, CONF_THRESH_LEAF - 0.02)
        CONF_THRESH_TRUNK = max(0.85, CONF_THRESH_TRUNK - 0.02)
        TOP2_GAP_MIN = max(0.12, TOP2_GAP_MIN - 0.02)
        SAM_TWO_STAGE = True
    elif THRESHOLD_PROFILE == "precision":
        # Stricter thresholds to reduce false positives
        LEAF_Q = min(0.95, LEAF_Q + 0.03)
        TRUNK_Q = min(0.96, TRUNK_Q + 0.04)
        LEAF_MIN_AREA = min(0.01, LEAF_MIN_AREA * 1.5)
        TRUNK_MIN_AREA = min(0.02, TRUNK_MIN_AREA * 1.5)
        CONF_THRESH_LEAF = min(0.98, CONF_THRESH_LEAF + 0.03)
        CONF_THRESH_TRUNK = min(0.99, CONF_THRESH_TRUNK + 0.03)
        TOP2_GAP_MIN = min(0.3, TOP2_GAP_MIN + 0.02)
        # Keep SAM_TWO_STAGE as configured


_apply_threshold_profile()


# Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Globals
model = None
cam = None
sam = None
predictor = None
transform = None
class_name_to_idx = {}
idx_to_class_name = []
num_classes = 0
total_processed = 0

# =========================
# Utilities
# =========================


def is_trunk_class(name: str) -> bool:
    n = name.lower()
    return ("trunk" in n) or ("bark" in n)


def class_policy(name: str):
    """Return (percentile_q, min_area_ratio, conf_thresh) for CAM thresholding & gating."""
    if is_trunk_class(name):
        return TRUNK_Q, TRUNK_MIN_AREA, CONF_THRESH_TRUNK
    else:
        return LEAF_Q, LEAF_MIN_AREA, CONF_THRESH_LEAF


def _resolve_best_model_path(checkpoint_root: str) -> str:
    """Find best_model*.pth; prefer smallest VL in name, else newest by mtime."""
    candidates = []
    for root, _, files in os.walk(checkpoint_root):
        for f in files:
            if f.lower().endswith(".pth") and f.startswith("best_model"):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(f"No best_model*.pth under: {checkpoint_root}")

    pick = None
    for fp in candidates:
        name = os.path.basename(fp)
        m_vl = re.search(r"VL([0-9]+(?:\.[0-9]+)?)", name)
        vl = float(m_vl.group(1)) if m_vl else None
        m_ep = re.search(r"epoch(\d+)", name)
        ep = int(m_ep.group(1)) if m_ep else None
        if pick is None:
            pick = (vl, ep, fp)
        else:
            if (vl is not None and (pick[0] is None or vl < pick[0])) or \
               (vl == pick[0] and ep is not None and pick[1] is not None and ep < pick[1]):
                pick = (vl, ep, fp)
    return pick[2] if pick else max(candidates, key=lambda p: os.path.getmtime(p))


def _resolve_classes_json(near_model_path: str, checkpoint_root: str) -> Optional[str]:
    model_dir = os.path.dirname(near_model_path)
    cand1 = os.path.join(model_dir, "training_stats", "classes.json")
    if os.path.isfile(cand1):
        return cand1
    cand2 = os.path.join(os.path.dirname(model_dir),
                         "training_stats", "classes.json")
    if os.path.isfile(cand2):
        return cand2
    found = []
    for root, _, files in os.walk(checkpoint_root):
        if 'training_stats' in os.path.basename(root).lower() and 'classes.json' in files:
            found.append(os.path.join(root, 'classes.json'))
    if found:
        found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return found[0]
    return None


def load_class_mapping():
    """Load class_name -> idx mapping from classes.json or folder names under base_dir."""
    global class_name_to_idx, idx_to_class_name, num_classes, classes_json
    if not classes_json and os.path.isdir(checkpoint_dir):
        classes_json = _resolve_classes_json(
            _resolve_best_model_path(checkpoint_dir), checkpoint_dir)

    if classes_json and os.path.isfile(classes_json):
        with open(classes_json, "r", encoding="utf-8") as f:
            class_name_to_idx = json.load(f)
        idxs = [None] * (max(class_name_to_idx.values()) + 1)
        for name, idx in class_name_to_idx.items():
            idxs[idx] = name
        idx_to_class_name = [c for c in idxs if c is not None]
        logger.info(f"Loaded class mapping from: {classes_json}")
    else:
        top_folders = [d for d in sorted(os.listdir(base_dir))
                       if os.path.isdir(os.path.join(base_dir, d))]
        class_name_to_idx = {name: i for i, name in enumerate(top_folders)}
        idx_to_class_name = top_folders
        logger.warning(
            "classes.json not found; using folder names from base_dir.")
    num_classes = len(idx_to_class_name)
    if num_classes == 0:
        raise RuntimeError("No classes found.")


def initialize_model_and_sam():
    """Create model matching checkpoint head, load weights, init GradCAM + SAM, set transforms."""
    global model, cam, sam, predictor, transform, num_classes

    model_path = _resolve_best_model_path(checkpoint_dir)
    state_all = torch.load(model_path, map_location="cpu")
    state = state_all["state_dict"] if (isinstance(
        state_all, dict) and "state_dict" in state_all) else state_all
    clean_state = OrderedDict((k[7:] if k.startswith(
        "module.") else k, v) for k, v in state.items())

    # infer #classes from classifier weight
    if "classifier.1.weight" in clean_state:
        ckpt_num_classes = clean_state["classifier.1.weight"].shape[0]
    else:
        lin_keys = [k for k in clean_state if k.endswith(
            ".weight") and "classifier" in k]
        if not lin_keys:
            raise RuntimeError(
                "Cannot infer number of classes from checkpoint.")
        ckpt_num_classes = clean_state[lin_keys[0]].shape[0]

    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, ckpt_num_classes))
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys: {unexpected}")
    if missing:
        logger.warning(f"Missing keys: {missing}")
    model.to(device).eval()
    logger.info(
        f"Loaded model: {model_path}  (classes in ckpt = {ckpt_num_classes})")

    # Make mapping consistent with checkpoint size
    if num_classes != ckpt_num_classes:
        logger.warning(
            f"[Mapping mismatch] mapping={num_classes} vs ckpt={ckpt_num_classes}; using ckpt size.")
        num_classes = ckpt_num_classes

    cam_target_layers = [model.features[-1]]
    global cam
    cam = GradCAM(model=model, target_layers=cam_target_layers)

    if not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
    global sam, predictor
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    global transform
    transform = transforms.Compose([
        transforms.Resize((CAM_INPUT_SIZE, CAM_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

# ---------- CAM helpers ----------


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    m, M = float(cam.min()), float(cam.max())
    if M > m:
        cam = (cam - m) / (M - m)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)
    return cam


def cam_to_clean_mask(cam_resized: np.ndarray, q: float, blur_sigma: float, min_area_ratio: float) -> np.ndarray:
    H, W = cam_resized.shape[:2]
    cam01 = normalize_cam(cam_resized)
    ksize = int(blur_sigma * 3) * 2 + 1
    cam_blur = cv2.GaussianBlur(cam01, (ksize, ksize), blur_sigma)
    t = float(np.quantile(cam_blur, q))
    binary = (cam_blur >= t).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.dilate(binary, k, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)
    min_area = max(1, int(min_area_ratio * H * W))
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return np.zeros((H, W), dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    return mask


def sample_points_within_contour(contour, num_points: int = 3):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return []
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    pts = contour.reshape(-1, 2)
    d = np.sum((pts - np.array([cx, cy]))**2, axis=1)
    if len(d) >= 2:
        i1 = int(np.argmax(d))
        d2 = d.copy()
        d2[i1] = -1
        i2 = int(np.argmax(d2))
        candidates = [(cx, cy), tuple(pts[i1]), tuple(pts[i2])]
    else:
        candidates = [(cx, cy)]
    return candidates[:max(1, num_points)]


def cam_with_tta(cam_obj, batch_tensor, targets):
    """Average CAMs with a horizontal flip if USE_CAM_TTA."""
    if not USE_CAM_TTA:
        return cam_obj(input_tensor=batch_tensor, targets=targets)
    cams0 = cam_obj(input_tensor=batch_tensor, targets=targets)
    x_flip = torch.flip(batch_tensor, dims=[3])
    cams_flip = cam_obj(input_tensor=x_flip, targets=targets)
    cams1 = np.stack([np.flip(c, axis=1) for c in cams_flip], axis=0)
    return 0.5 * (cams0 + cams1)

# ---------- Mask -> YOLO polygon ----------


def mask_to_yolo_polygon(mask_path, class_id, save_txt_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return
    binary = (mask == class_id).astype(np.uint8)
    if np.sum(binary) == 0:
        return
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
    os.makedirs(os.path.dirname(save_txt_path), exist_ok=True)
    with open(save_txt_path, "w", encoding="utf-8") as f:
        for contour in contours:
            if len(contour) < 3:
                continue
            poly = contour.squeeze().astype(float)
            if poly.ndim == 1:
                poly = poly.reshape(-1, 2)
            poly[:, 0] /= w
            poly[:, 1] /= h
            coords = " ".join(f"{c:.6f}" for c in poly.flatten().tolist())
            f.write(f"{class_id} {coords}\n")

# ---------- Directory discovery ----------


def discover_class_folders() -> List[Tuple[str, List[Tuple[List[str], str]]]]:
    """
    Returns: list of (class_name, [ (image_paths_in_subdir, rel_subdir), ... ])
    rel_subdir is relative to <base_dir>/<class_name> ("" for class root).
    """
    result = []
    top_classes = [d for d in sorted(os.listdir(base_dir))
                   if os.path.isdir(os.path.join(base_dir, d))]
    for cls_name in top_classes:
        class_root = os.path.join(base_dir, cls_name)
        items = []
        for root, _, files in os.walk(class_root):
            rel = os.path.relpath(root, class_root)
            rel = "" if rel == "." else rel
            img_paths = [os.path.join(root, f)
                         for f in files if f.lower().endswith(image_exts)]
            if img_paths:
                items.append((img_paths, rel))
        if items:
            result.append((cls_name, items))
    return result

# ---------- Core processing ----------


def process_images_in_batch(image_paths: List[str],
                            target_class: int,
                            class_name: str,
                            save_root_mask: str,
                            save_root_labels: str):
    """CAM -> class-aware binary -> stable points -> SAM -> save mask + polygon."""
    global total_processed
    try:
        q_thr, min_area_ratio, conf_thr = class_policy(class_name)

        # load and stack
        batch_imgs, originals = [], []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            originals.append((p, img))
            batch_imgs.append(transform(img).unsqueeze(0))
        batch_tensor = torch.cat(batch_imgs, dim=0).to(device)

        # classifier confidence (gate)
        with torch.no_grad():
            logits = model(batch_tensor)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            top2 = np.partition(-probs, 1, axis=1)[:, :2]
            top2 = -top2
            gaps = top2[:, 0] - top2[:, 1]
            class_probs = probs[:, target_class]
        keep = (class_probs >= conf_thr) & (gaps >= TOP2_GAP_MIN)
        if not np.any(keep):
            return
        idxs = np.where(keep)[0].tolist()
        batch_tensor = batch_tensor[idxs]
        originals = [originals[i] for i in idxs]
        class_probs = class_probs[idxs]

        # CAM (with optional TTA)
        targets = [ClassifierOutputTarget(target_class)] * len(originals)
        if device.type == 'cuda' and USE_AMP:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                grayscale_cams = cam_with_tta(cam, batch_tensor, targets)
        else:
            grayscale_cams = cam_with_tta(cam, batch_tensor, targets)

        for k, (img_path, original_image) in enumerate(originals):
            cam_small = grayscale_cams[k].astype(np.float32)
            cam_resized = cv2.resize(
                cam_small, original_image.size, interpolation=cv2.INTER_LINEAR)

            # class-aware mask
            clean_mask = cam_to_clean_mask(
                cam_resized, q=q_thr, blur_sigma=BLUR_SIGMA, min_area_ratio=min_area_ratio)
            if clean_mask.sum() == 0:
                continue
            contours, _ = cv2.findContours(
                clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            # largest component
            contour = max(contours, key=cv2.contourArea)
            pts = sample_points_within_contour(
                contour, num_points=PROMPT_POINTS)
            if not pts:
                continue

            predictor.set_image(np.array(original_image))
            if SAM_TWO_STAGE:
                masks, scores, logits = predictor.predict(
                    point_coords=np.array(pts, dtype=np.float32),
                    point_labels=np.ones(len(pts), dtype=np.int32),
                    multimask_output=True
                )
                best = int(np.argmax(scores))
                best_mask_input = logits[best, :, :]
                refined_mask, _, _ = predictor.predict(
                    point_coords=np.array(pts, dtype=np.float32),
                    point_labels=np.ones(len(pts), dtype=np.int32),
                    mask_input=best_mask_input[None, :, :],
                    multimask_output=False
                )
                refined_mask = np.asarray(refined_mask).squeeze().astype(bool)
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(pts, dtype=np.float32),
                    point_labels=np.ones(len(pts), dtype=np.int32),
                    multimask_output=False
                )
                refined_mask = np.asarray(masks).squeeze().astype(bool)

            final_mask = np.full(refined_mask.shape,
                                 Background_value, dtype=np.uint8)
            final_mask[refined_mask] = np.uint8(target_class)

            # save paths
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            os.makedirs(save_root_mask, exist_ok=True)
            mask_path = os.path.join(save_root_mask, f"mask_{base_name}.png")
            cv2.imwrite(mask_path, final_mask)

            os.makedirs(save_root_labels, exist_ok=True)
            yolo_txt = os.path.join(save_root_labels, f"{base_name}.txt")
            mask_to_yolo_polygon(mask_path, target_class, yolo_txt)

            total_processed += 1
            if total_processed % LOG_EVERY_N == 0:
                logger.info(
                    f"[{class_name}] {total_processed} | conf={class_probs[k]:.2f} -> {yolo_txt}")

    except Exception as e:
        logger.error(f"Batch error: {e}", exc_info=False)
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def process_all_folders():
    load_class_mapping()
    initialize_model_and_sam()

    for cls_name, sublists in discover_class_folders():
        if cls_name not in class_name_to_idx:
            logger.warning(
                f"Folder '{cls_name}' not in class mapping; skipping.")
            continue
        cls_idx = int(class_name_to_idx[cls_name])
        logger.info(f"\nClass: {cls_name} (idx={cls_idx})")

        for img_paths, rel in sublists:
            mask_root = os.path.join(base_dir, f"{cls_name}_mask",   rel) if rel else os.path.join(
                base_dir, f"{cls_name}_mask")
            label_root = os.path.join(base_dir, f"{cls_name}_labels", rel) if rel else os.path.join(
                base_dir, f"{cls_name}_labels")
            # batched
            for i in range(0, len(img_paths), BATCH_SIZE):
                batch = img_paths[i:i+BATCH_SIZE]
                process_images_in_batch(
                    batch, cls_idx, cls_name, mask_root, label_root)

    logger.info("Done.")


# ---------- Entry ----------
if __name__ == "__main__":
    try:
        process_all_folders()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
