import os
import json
import random
import logging
from collections import OrderedDict
from typing import Optional

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
base_dir = "E:/Santosh_master_thesis/LOT_all_images_labeled"

# Checkpoint folder produced by your training script
checkpoint_dir = "E:/Santosh_master_thesis/Checkpoints_labeled_LOT"
# We'll resolve the actual best model path dynamically (supports names like best_model_epoch12_VL0.38.pth)
model_path = None  # resolved at runtime
# resolved at runtime, prefers alongside the chosen model (../training_stats/classes.json)
classes_json = None

# Segment-Anything checkpoint
sam_checkpoint = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"

# Processing params
Threshold_value = 150         # CAM threshold to make binary mask
# positive points per contour for SAM refinement (reduced for speed)
No_of_sampled_points = 2
# inference batch size for CAM (tune based on VRAM)
Batch_size = 24
# background fill in saved mask (must not collide with class IDs 0..N-1)
Background_value = 255

# Speed/quality knobs
CAM_INPUT_SIZE = 384          # CAM input resolution (smaller = faster)
USE_AMP = True                # mixed precision for CAM generation
SAM_TWO_STAGE = False         # if False, run single-pass SAM (faster)
MAX_CONTOURS_PER_IMAGE = 5    # keep only top-N largest CAM contours per image
MIN_CONTOUR_AREA_RATIO = 0.0005  # ignore tiny contours (<0.05% of image area)
LIMIT_IMAGES_PER_CLASS = None    # e.g., 500 for quick runs; None to process all
LOG_EVERY_N = 25                 # info log frequency for saved masks/polygons

# I/O
image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif',
              '.tiff', '.JPG', '.PNG', '.JPEG')

# Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Globals (initialized later)
model = None
cam = None
sam = None
predictor = None
transform = None
class_name_to_idx = {}
idx_to_class_name = []
num_classes = 0
total_processed = 0  # throttled logging counter

# =========================
# Utils
# =========================


def load_class_mapping():
    """
    Loads class_name -> index mapping from classes.json produced by training.
    Falls back to the folder names in base_dir if classes.json is missing.
    """
    global class_name_to_idx, idx_to_class_name, num_classes, classes_json

    # If classes_json wasn't set yet, try to discover one under checkpoint_dir
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
        # Build ordered list by index
        idx_to_class_name = [None] * len(class_name_to_idx)
        for name, idx in class_name_to_idx.items():
            if idx < 0:
                continue
            if idx >= len(idx_to_class_name):
                idx_to_class_name.extend(
                    [None] * (idx - len(idx_to_class_name) + 1))
            idx_to_class_name[idx] = name
        # Compact any None (in case of sparse idx)
        idx_to_class_name = [c for c in idx_to_class_name if c is not None]
        logger.info(f"Loaded class mapping from: {classes_json}")
    else:
        # Fallback: infer from folders under base_dir (sorted)
        top_folders = [d for d in sorted(os.listdir(base_dir))
                       if os.path.isdir(os.path.join(base_dir, d))]
        class_name_to_idx = {name: i for i, name in enumerate(top_folders)}
        idx_to_class_name = top_folders
        logger.warning("classes.json not found; using folder names as class mapping: "
                       f"{class_name_to_idx}")

    num_classes = len(idx_to_class_name)
    if num_classes == 0:
        raise RuntimeError("No classes found. Check base_dir or classes.json.")


def _resolve_best_model_path(checkpoint_root: str) -> str:
    """
    Find the best model checkpoint under checkpoint_root.
    Preference order:
      1) Files matching best_model_epoch*_VL*.pth with the lowest VL value.
      2) Any best_model*.pth, most recent by mtime.
    Searches recursively. Returns absolute path or raises FileNotFoundError.
    """
    candidates = []
    for root, _, files in os.walk(checkpoint_root):
        for f in files:
            if f.lower().endswith('.pth') and f.startswith('best_model'):
                full = os.path.join(root, f)
                candidates.append(full)

    if not candidates:
        raise FileNotFoundError(
            f"No best_model*.pth found under: {checkpoint_root}")

    import re
    with_scores = []  # (vl_score or None, epoch or None, fullpath)
    for fp in candidates:
        name = os.path.basename(fp)
        # Extract VL value if present, e.g., best_model_epoch12_VL0.38.pth
        m_vl = re.search(r"VL([0-9]+(?:\.[0-9]+)?)", name)
        vl = float(m_vl.group(1)) if m_vl else None
        m_ep = re.search(r"epoch(\d+)", name)
        ep = int(m_ep.group(1)) if m_ep else None
        with_scores.append((vl, ep, fp))

    # Prefer ones with VL, choose smallest VL; tie-breaker: lower epoch; fallback: mtime
    with_vl = [t for t in with_scores if t[0] is not None]
    if with_vl:
        with_vl.sort(key=lambda t: (t[0], t[1] if t[1] is not None else 1e9))
        return with_vl[0][2]

    # No VL annotated; choose most recent by mtime
    latest = max(candidates, key=lambda p: os.path.getmtime(p))
    return latest


def _resolve_classes_json(near_model_path: str, checkpoint_root: str) -> Optional[str]:
    """
    Try to locate classes.json. Preference order:
      1) ../training_stats/classes.json relative to the model directory.
      2) ../../training_stats/classes.json (one more level up).
      3) Any training_stats/classes.json found recursively under checkpoint_root, pick most recent.
    Returns path or None if not found.
    """
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
    """
    Initializes:
      - EfficientNet-V2-S classifier with best_model.pth weights
      - GradCAM for CAM generation
      - SAM+Predictor for mask refinement
      - Preprocessing transform
    """
    global model, cam, sam, predictor, transform

    # Resolve checkpoint file dynamically
    global model_path, classes_json
    try:
        model_path = _resolve_best_model_path(checkpoint_dir)
    except FileNotFoundError as e:
        raise FileNotFoundError(str(e))

    # Model
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, num_classes))

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    state = torch.load(model_path, map_location="cpu")
    # Accept both raw state_dict or checkpoint dict
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Remove DistributedDataParallel 'module.' prefix if present
    new_state = OrderedDict()
    for k, v in state.items():
        k2 = k[7:] if k.startswith("module.") else k
        new_state[k2] = v

    model.load_state_dict(new_state, strict=False)
    model.to(device).eval()
    logger.info(f"Model loaded from: {model_path}")

    # Resolve classes.json now that we know where the model is
    if classes_json is None:
        classes_json_candidate = _resolve_classes_json(
            model_path, checkpoint_dir)
        if classes_json_candidate and os.path.isfile(classes_json_candidate):
            classes_json = classes_json_candidate
        else:
            classes_json = None

    # GradCAM on the last feature block
    cam_target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=cam_target_layers)

    # SAM
    if not os.path.isfile(sam_checkpoint):
        raise FileNotFoundError(f"SAM checkpoint not found: {sam_checkpoint}")
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    # Preprocessing (match training normalization)
    transform = transforms.Compose([
        # fixed resolution for CAM input
        transforms.Resize((CAM_INPUT_SIZE, CAM_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])


def sample_points_within_contour(contour, num_points):
    rect = cv2.boundingRect(contour)
    mask = np.zeros((rect[3], rect[2]), dtype=np.uint8)
    shifted = contour - np.array([[rect[0], rect[1]]])
    cv2.drawContours(mask, [shifted], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(mask == 255)
    if len(xs) == 0:
        return []
    if len(xs) <= num_points:
        return [(int(xs[i] + rect[0]), int(ys[i] + rect[1])) for i in range(len(xs))]
    sel = random.sample(range(len(xs)), num_points)
    return [(int(xs[i] + rect[0]), int(ys[i] + rect[1])) for i in sel]


def mask_to_yolo_polygon(mask_path, class_id, save_txt_path):
    """
    Convert a single-class mask image to YOLO polygon format:
      class_id x1 y1 x2 y2 ... (normalized coords)
    One line per connected component (contour).
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


def process_images_in_batch(image_paths, target_class, threshold_value, num_sampled_points, save_folder):
    """
    Generate CAM -> threshold -> sample points -> SAM refine -> save mask + YOLO polygon.
    """
    try:
        global model, cam, predictor, transform, device

        # Load images
        batch_imgs = []
        originals = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            originals.append((p, img))
            batch_imgs.append(transform(img).unsqueeze(0))
        batch_tensor = torch.cat(batch_imgs, dim=0).to(device)

        # CAM for the target class
        targets = [ClassifierOutputTarget(target_class)] * len(image_paths)
        if device.type == 'cuda' and USE_AMP:
            # New API (avoids deprecation warning)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                grayscale_cams = cam(
                    input_tensor=batch_tensor, targets=targets)
        else:
            grayscale_cams = cam(input_tensor=batch_tensor, targets=targets)

        # Process each image
        for idx, (img_path, original_image) in enumerate(originals):
            grayscale_cam = grayscale_cams[idx]
            cam_resized = cv2.resize(
                grayscale_cam, original_image.size, interpolation=cv2.INTER_LINEAR)

            # Threshold CAM to binary
            _, binary_map = cv2.threshold(
                np.uint8(255 * cam_resized), threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(
                binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                logger.info(
                    f"No activation contours for {img_path}; skipping mask.")
                continue

            # Filter small contours and keep only largest N by area
            img_w, img_h = original_image.size
            min_area = max(1, int(MIN_CONTOUR_AREA_RATIO * img_w * img_h))
            contours = [c for c in contours if cv2.contourArea(c) >= min_area]
            if not contours:
                logger.info(
                    f"Only tiny activations for {img_path}; skipping mask.")
                continue
            contours = sorted(contours, key=lambda c: cv2.contourArea(
                c), reverse=True)[:MAX_CONTOURS_PER_IMAGE]

            # Points for SAM
            all_points, all_labels = [], []
            for contour in contours:
                pts = sample_points_within_contour(contour, num_sampled_points)
                if pts:
                    all_points.extend(pts)
                    all_labels.extend([1] * len(pts))  # positive points

            if not all_points:
                logger.info(
                    f"No points sampled for {img_path}; skipping mask.")
                continue

            # SAM inference (single-pass by default for speed)
            predictor.set_image(np.array(original_image))
            if SAM_TWO_STAGE:
                masks, scores, logits = predictor.predict(
                    point_coords=np.array(all_points, dtype=np.float32),
                    point_labels=np.array(all_labels, dtype=np.int32),
                    multimask_output=True
                )
                best_idx = int(np.argmax(scores))
                best_mask_input = logits[best_idx, :, :]
                refined_mask, _, _ = predictor.predict(
                    point_coords=np.array(all_points, dtype=np.float32),
                    point_labels=np.array(all_labels, dtype=np.int32),
                    mask_input=best_mask_input[None, :, :],
                    multimask_output=False
                )
                refined_mask = np.squeeze(refined_mask).astype(bool)
            else:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array(all_points, dtype=np.float32),
                    point_labels=np.array(all_labels, dtype=np.int32),
                    multimask_output=False
                )
                if isinstance(scores, np.ndarray) and scores.size > 1:
                    best_idx = int(np.argmax(scores))
                    refined_mask = np.asarray(masks[best_idx]).astype(bool)
                else:
                    refined_mask = np.asarray(masks).squeeze().astype(bool)

            # Compose final mask with target class id
            final_mask = np.full(refined_mask.shape,
                                 Background_value, dtype=np.uint8)
            final_mask[refined_mask] = np.uint8(target_class)

            # Save mask next to class save folder mirroring subfolder structure
            os.makedirs(save_folder, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_save_path = os.path.join(save_folder, f"mask_{base_name}.png")
            cv2.imwrite(mask_save_path, final_mask)

            # Save YOLO polygon
            yolo_txt = mask_save_path.replace(".png", ".txt")
            mask_to_yolo_polygon(mask_save_path, target_class, yolo_txt)

            # Throttled logging
            global total_processed
            total_processed += 1
            if total_processed % LOG_EVERY_N == 0:
                logger.info(
                    f"Processed {total_processed} masks... last: {mask_save_path}")

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
    # Load mapping, init model+sam once
    load_class_mapping()
    initialize_model_and_sam()

    class_sets = discover_class_folders()

    # Process each class folder using the class index from training mapping
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
        for img_paths, save_folder in path_sets:
            # batch over this subfolder
            # Optionally limit total images per class for faster runs
            if LIMIT_IMAGES_PER_CLASS is not None and processed_count >= LIMIT_IMAGES_PER_CLASS:
                break
            for i in range(0, len(img_paths), Batch_size):
                if LIMIT_IMAGES_PER_CLASS is not None and processed_count >= LIMIT_IMAGES_PER_CLASS:
                    break
                batch = img_paths[i:i + Batch_size]
                process_images_in_batch(
                    batch, cls_idx, Threshold_value, No_of_sampled_points, save_folder)
                processed_count += len(batch)

    logger.info("Done.")


if __name__ == "__main__":
    try:
        process_all_folders()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
