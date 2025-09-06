# -*- coding: utf-8 -*-
"""
CAM -> SAM mask & YOLO polygon exporter
- Robust CAM post-processing for clean single-object masks
- Stable SAM prompts (centroid + farthest points)
- Saves masks to <species>_mask/
- Saves YOLO label .txt to <species>_labels/ (mirrors subfolders)
- Label filenames DO NOT include 'mask_' prefix

"""

import os
import json
import random
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
logger = logging.getLogger(__name__)

# =========================
# Config
# =========================
# Root of the classified images (top-level folder per class, subfolders allowed)
base_dir = r"E:/Santosh_master_thesis/LT_species_organ_10_species"

# Checkpoint folder produced by your training script
checkpoint_dir = r"E:/Santosh_master_thesis/Checkpoints_species_organ_weighted_random_sampler_focal_loss"
model_path = None          # resolved at runtime
classes_json = None        # resolved at runtime

# Segment-Anything checkpoint
sam_checkpoint = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"

# Processing knobs
# CAM post-processing (robust defaults)
CAM_INPUT_SIZE = 384
CAM_PERCENTILE = 0.85       # 0.80–0.90 usually good
BLUR_SIGMA = 3.0            # Gaussian blur sigma (px)
MIN_CONTOUR_AREA_RATIO = 0.002  # drop tiny blobs (<0.2% image)
No_of_sampled_points = 3    # centroid + farthest two points
USE_AMP = True
SAM_TWO_STAGE = False
Batch_size = 24

# I/O and mask encoding
Background_value = 255
image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif',
              '.tiff', '.JPG', '.PNG', '.JPEG')

# Force generation: if True, do not skip any image due to weak/empty CAM.
# Fallback to center-based prompts and, if SAM returns empty, fill the whole image.
FORCE_GENERATE_ALL = True

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
total_processed = 0  # throttled logging counter
LOG_EVERY_N = 25

# =========================
# Utils
# =========================


def load_class_mapping():
    """Loads class_name -> index mapping from classes.json; fallback to folder names."""
    global class_name_to_idx, idx_to_class_name, num_classes, classes_json

    # discover classes.json if not set
    if not classes_json and os.path.isdir(checkpoint_dir):
        discovered = []
        for root, dirs, files in os.walk(checkpoint_dir):
            if 'training_stats' in os.path.basename(root).lower() and 'classes.json' in files:
                discovered.append(os.path.join(root, 'classes.json'))
        if discovered:
            discovered.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            classes_json = discovered[0]

    if classes_json and os.path.isfile(classes_json):
        with open(classes_json, "r", encoding="utf-8") as f:
            class_name_to_idx = json.load(f)  # {name: idx}
        idxs = [None] * (max(class_name_to_idx.values()) + 1)
        for name, idx in class_name_to_idx.items():
            idxs[idx] = name
        idx_to_class_name = [c for c in idxs if c is not None]
        logger.info(f"Loaded class mapping from: {classes_json}")
    else:
        # fallback to folder names in base_dir
        top_folders = [d for d in sorted(os.listdir(base_dir))
                       if os.path.isdir(os.path.join(base_dir, d))]
        class_name_to_idx = {name: i for i, name in enumerate(top_folders)}
        idx_to_class_name = top_folders
        logger.warning(
            "classes.json not found; using folder names as class mapping.")

    num_classes = len(idx_to_class_name)
    if num_classes == 0:
        raise RuntimeError("No classes found. Check base_dir or classes.json.")


def _resolve_best_model_path(checkpoint_root: str) -> str:
    """Return best_model*.pth under checkpoint_root (prefers lowest VL)."""
    candidates = []
    for root, _, files in os.walk(checkpoint_root):
        for f in files:
            if f.lower().endswith('.pth') and f.startswith('best_model'):
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(
            f"No best_model*.pth found under: {checkpoint_root}")

    import re
    best = None
    for fp in candidates:
        name = os.path.basename(fp)
        m_vl = re.search(r"VL([0-9]+(?:\.[0-9]+)?)", name)
        vl = float(m_vl.group(1)) if m_vl else None
        m_ep = re.search(r"epoch(\d+)", name)
        ep = int(m_ep.group(1)) if m_ep else None
        if best is None:
            best = (vl, ep, fp)
        else:
            # choose smallest VL; tie by lower epoch; else keep previous
            if (vl is not None and (best[0] is None or vl < best[0])) or \
               (vl == best[0] and ep is not None and best[1] is not None and ep < best[1]):
                best = (vl, ep, fp)

    if best and best[2]:
        return best[2]
    # fallback: newest by mtime
    return max(candidates, key=lambda p: os.path.getmtime(p))


def _resolve_classes_json(near_model_path: str, checkpoint_root: str) -> Optional[str]:
    """Locate a training_stats/classes.json near the model or anywhere under checkpoint_root."""
    model_dir = os.path.dirname(near_model_path)
    cand1 = os.path.join(model_dir, 'training_stats', 'classes.json')
    if os.path.isfile(cand1):
        return cand1
    parent = os.path.dirname(model_dir)
    cand2 = os.path.join(parent, 'training_stats', 'classes.json')
    if os.path.isfile(cand2):
        return cand2
    found = []
    for root, dirs, files in os.walk(checkpoint_root):
        if 'training_stats' in os.path.basename(root).lower() and 'classes.json' in files:
            found.append(os.path.join(root, 'classes.json'))
    if found:
        found.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return found[0]
    return None


def initialize_model_and_sam():
    """Init classifier + GradCAM + SAM + transforms, making head size match checkpoint."""
    global model, cam, sam, predictor, transform, model_path, classes_json, num_classes

    model_path = _resolve_best_model_path(checkpoint_dir)

    # ---- read checkpoint + infer #classes from saved head ----
    state_all = torch.load(model_path, map_location="cpu")
    state = state_all["state_dict"] if (isinstance(
        state_all, dict) and "state_dict" in state_all) else state_all

    # Strip possible 'module.' prefix
    clean_state = OrderedDict((k[7:] if k.startswith(
        "module.") else k, v) for k, v in state.items())

    # Infer number of classes from the saved classifier weight/bias
    # (works for EfficientNet-V2-S: classifier[1] is the Linear layer)
    if "classifier.1.weight" in clean_state:
        ckpt_num_classes = clean_state["classifier.1.weight"].shape[0]
    else:
        # Fallback: try common alt keys
        lin_keys = [k for k in clean_state if k.endswith(
            ".weight") and "classifier" in k]
        if not lin_keys:
            raise RuntimeError(
                "Could not infer number of classes from checkpoint.")
        ckpt_num_classes = clean_state[lin_keys[0]].shape[0]

    # ---- build model with checkpoint head size ----
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, ckpt_num_classes))

    # now load weights
    missing, unexpected = model.load_state_dict(clean_state, strict=False)
    if unexpected:
        logger.warning(f"Unexpected keys in state_dict: {unexpected}")
    if missing:
        logger.warning(f"Missing keys when loading state_dict: {missing}")
    model.to(device).eval()
    logger.info(
        f"Model loaded from: {model_path} (ckpt classes = {ckpt_num_classes})")

    # ---- resolve classes.json near model, and sanity-check mapping size ----
    if classes_json is None:
        classes_json_candidate = _resolve_classes_json(
            model_path, checkpoint_dir)
        classes_json = classes_json_candidate if classes_json_candidate else None

    # If you already called load_class_mapping() earlier, its num_classes may be wrong.
    # Re-load mapping now and check length vs ckpt.
    try:
        load_class_mapping()
        if num_classes != ckpt_num_classes:
            logger.warning(f"[Mapping mismatch] classes.json/base_dir has {num_classes} classes "
                           f"but checkpoint has {ckpt_num_classes}. "
                           f"Proceeding with checkpoint size; labels will use checkpoint indices.")
            num_classes = ckpt_num_classes  # make downstream code consistent
    except Exception as e:
        logger.warning(f"Could not (re)load class mapping here: {e}. "
                       f"Continuing with ckpt_num_classes={ckpt_num_classes}.")
        num_classes = ckpt_num_classes

    # ---- GradCAM target layer ----
    cam_target_layers = [model.features[-1]]
    global cam
    cam = GradCAM(model=model, target_layers=cam_target_layers)

    # ---- SAM ----
    if not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
    global sam, predictor
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # ---- preprocessing ----
    global transform
    transform = transforms.Compose([
        transforms.Resize((CAM_INPUT_SIZE, CAM_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


# ---------- CAM post-processing helpers ----------

def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    m, M = float(cam.min()), float(cam.max())
    if M > m:
        cam = (cam - m) / (M - m)
    else:
        cam = np.zeros_like(cam, dtype=np.float32)
    return cam


def cam_to_clean_mask(cam_resized: np.ndarray,
                      q: float = CAM_PERCENTILE,
                      blur_sigma: float = BLUR_SIGMA,
                      min_area_ratio: float = MIN_CONTOUR_AREA_RATIO) -> np.ndarray:
    """Normalize -> blur -> percentile threshold -> morphology -> keep largest component."""
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
    """Return stable prompts: centroid + two farthest boundary points (max 3)."""
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


def mask_to_yolo_polygon(mask_path, class_id, save_txt_path):
    """
    Convert a single-class mask image to YOLO polygon format:
      class_id x1 y1 x2 y2 ... (normalized coords)  ; one line per component
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.warning(f"Could not read mask: {mask_path}")
        return

    binary = (mask == class_id).astype(np.uint8)
    if np.sum(binary) == 0:
        return

    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]
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
    Returns: list of tuples (class_name, [ (image_paths_in_subdir, rel_subdir), ... ])
    rel_subdir is the path relative to <base_dir>/<class_name> ("" for root).
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
    """CAM -> clean binary -> sample points -> SAM -> save mask + YOLO polygon (labels without 'mask_' prefix)."""
    try:
        global model, cam, predictor, transform, device, total_processed

        # load + preprocess batch for CAM
        batch_imgs = []
        originals = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            originals.append((p, img))
            batch_imgs.append(transform(img).unsqueeze(0))
        batch_tensor = torch.cat(batch_imgs, dim=0).to(device)

        targets = [ClassifierOutputTarget(target_class)] * len(image_paths)
        if device.type == 'cuda' and USE_AMP:
            with torch.amp.autocast('cuda', dtype=torch.float16):
                grayscale_cams = cam(
                    input_tensor=batch_tensor, targets=targets)
        else:
            grayscale_cams = cam(input_tensor=batch_tensor, targets=targets)

        for idx, (img_path, original_image) in enumerate(originals):
            # [0..1] float from pytorch-grad-cam
            cam_small = grayscale_cams[idx]
            cam_resized = cv2.resize(
                cam_small.astype(np.float32),
                original_image.size,
                interpolation=cv2.INTER_LINEAR,
            )

            # robust CAM -> single clean mask
            clean_mask = cam_to_clean_mask(cam_resized)
            contours, _ = cv2.findContours(
                clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pts = None
            if contours:
                # largest contour only
                contours = [max(contours, key=cv2.contourArea)]
                pts = sample_points_within_contour(
                    contours[0], num_points=No_of_sampled_points)
            if (not contours or not pts) and FORCE_GENERATE_ALL:
                # Fallback: prompt SAM at center and two diagonal near-corners
                w, h = original_image.size
                cx, cy = max(1, w // 2), max(1, h // 2)
                dx, dy = max(2, w // 6), max(2, h // 6)
                pts = [(cx, cy), (dx, dy), (w - dx, h - dy)]
            if not pts:
                logger.info(f"No usable prompts for {img_path}; skipping.")
                continue

            predictor.set_image(np.array(original_image))

            if SAM_TWO_STAGE:
                masks, scores, logits = predictor.predict(
                    point_coords=np.array(pts, dtype=np.float32),
                    point_labels=np.ones(len(pts), dtype=np.int32),
                    multimask_output=True,
                )
                best_idx = int(np.argmax(scores))
                best_mask_input = logits[best_idx, :, :]
                refined_mask, _, _ = predictor.predict(
                    point_coords=np.array(pts, dtype=np.float32),
                    point_labels=np.ones(len(pts), dtype=np.int32),
                    mask_input=best_mask_input[None, :, :],
                    multimask_output=False,
                )
                refined_mask = np.squeeze(refined_mask).astype(bool)
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(pts, dtype=np.float32),
                    point_labels=np.ones(len(pts), dtype=np.int32),
                    multimask_output=False,
                )
                if isinstance(scores, np.ndarray) and scores.size > 1:
                    best_idx = int(np.argmax(scores))
                    refined_mask = np.asarray(masks[best_idx]).astype(bool)
                else:
                    refined_mask = np.asarray(masks).squeeze().astype(bool)

            # If SAM produced an empty mask and forcing is enabled, fill whole image
            if not refined_mask.any() and FORCE_GENERATE_ALL:
                refined_mask = np.ones_like(refined_mask, dtype=bool)

            # compose final mask (class id inside; 255 elsewhere)
            final_mask = np.full(refined_mask.shape,
                                 Background_value, dtype=np.uint8)
            final_mask[refined_mask] = np.uint8(target_class)

            # -------- save paths (mask + labels sibling folders) --------
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            # mask keeps 'mask_' prefix
            mask_dir = save_root_mask
            os.makedirs(mask_dir, exist_ok=True)
            mask_save_path = os.path.join(mask_dir, f"mask_{base_name}.png")
            cv2.imwrite(mask_save_path, final_mask)

            # label has NO prefix; save under <species>_labels / same subfolder
            label_dir = save_root_labels
            os.makedirs(label_dir, exist_ok=True)
            yolo_txt = os.path.join(label_dir, f"{base_name}.txt")
            mask_to_yolo_polygon(mask_save_path, target_class, yolo_txt)

            total_processed += 1
            if total_processed % LOG_EVERY_N == 0:
                logger.info(
                    f"Processed {total_processed} ... last mask: {mask_save_path} | label: {yolo_txt}"
                )

    except Exception as e:
        logger.error(f"Batch error: {e}", exc_info=False)
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def discover_class_folders():
    """
    Returns list of (class_name, [(image_path, save_folder), ...])
    Save folders are created under <base_dir>/<class_name>_mask mirroring subfolders.
    """
    result = []
    top_classes = [d for d in sorted(os.listdir(base_dir))
                   if os.path.isdir(os.path.join(base_dir, d))]
    for cls_name in top_classes:
        class_root = os.path.join(base_dir, cls_name)
        save_root = os.path.join(base_dir, f"{cls_name}_mask")
        items = []
        for root, _, files in os.walk(class_root):
            rel = os.path.relpath(root, class_root)
            save_folder = os.path.join(
                save_root, rel) if rel != "." else save_root
            img_paths = [os.path.join(root, f)
                         for f in files if f.lower().endswith(image_exts)]
            if img_paths:
                items.append((img_paths, save_folder))
        if items:
            result.append((cls_name, items))
    return result


def process_all_folders():
    load_class_mapping()
    try:
        load_class_mapping()
    except Exception as e:
        logger.warning(f"Class mapping not loaded at start: {e}")

    # sets num_classes to match checkpoint, and re-checks mapping
    initialize_model_and_sam()

    class_sets = discover_class_folders()

    for cls_name, path_sets in class_sets:
        if cls_name not in class_name_to_idx:
            logger.warning(
                f"Folder '{cls_name}' not in training classes, skipping.")
            continue
        cls_idx = int(class_name_to_idx[cls_name])
        logger.info(f"\nClass: {cls_name} (idx: {cls_idx})")

        total_imgs = sum(len(paths) for paths, _ in path_sets)
        logger.info(f"Total images found: {total_imgs}")

        processed_count = 0
        for img_paths, rel_subdir in path_sets:
            # sibling output roots for this subdir
            mask_root = os.path.join(base_dir, f"{cls_name}_mask", rel_subdir) if rel_subdir else os.path.join(
                base_dir, f"{cls_name}_mask")
            label_root = os.path.join(base_dir, f"{cls_name}_labels", rel_subdir) if rel_subdir else os.path.join(
                base_dir, f"{cls_name}_labels")

            # batching
            for i in range(0, len(img_paths), Batch_size):
                batch = img_paths[i:i + Batch_size]
                process_images_in_batch(
                    batch,
                    cls_idx,
                    cls_name,
                    save_root_mask=mask_root,
                    save_root_labels=label_root
                )
                processed_count += len(batch)

    logger.info("Done.")

# ---------- Entry ----------


if __name__ == "__main__":
    try:
        process_all_folders()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
