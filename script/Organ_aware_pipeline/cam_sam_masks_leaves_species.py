import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segment_anything import sam_model_registry, SamPredictor
import random
import torch.nn as nn
from collections import OrderedDict
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup base directory and parameters
base_dir = "E:/Santosh_master_thesis/classified_Leaves"

Threshold_value = 60  # Medium focused with details
No_of_sampled_points = 2
No_classes = 10
Batch_size = 32
Background_class = 10

model_path = "E:/Santosh_master_thesis/Checkpoints_efficientnet_leaves/best_model_swa_final_0.12.pth"
sam_checkpoint = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"

patterns = tuple(['.jpg', '.png', '.JPEG', '.JPG', '.PNG', '.jpeg'])

# --- Acceptance/cleanup thresholds ---
# Keep only largest component; accept if it's tall and big enough.
LARGEST_MIN_HEIGHT_FRAC = 0.70   # bbox height >= 70% of image height
LARGEST_MIN_AREA_FRAC = 0.10    # component area >= 10% of image area
# Cleanup thresholds
HOLE_FILL_MAX_FRAC = 0.005   # fill holes up to 0.5% of image area
# remove components smaller than 0.3% of image area
SPECKLE_REMOVE_MAX_FRAC = 0.01
# If a second component is big enough compared to largest, split/save separately
SECOND_MIN_REL_AREA = 0.20    # second area >= 20% of largest area


def _remove_small_components(mask01: np.ndarray, max_frac: float) -> np.ndarray:
    """Remove connected components whose area is less than max_frac of image area."""
    h, w = mask01.shape[:2]
    min_area = max(1, int(max_frac * h * w))
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask01, dtype=np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 1
    return out


def _fill_small_holes(mask01: np.ndarray, hole_max_frac: float) -> np.ndarray:
    """Fill holes inside the foreground up to hole_max_frac of image area."""
    h, w = mask01.shape[:2]
    inv = (mask01 == 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = mask01.copy().astype(np.uint8)
    limit = max(1, int(hole_max_frac * h * w))
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= limit:
            out[labels == i] = 1
    return out


def _connected_components(mask01: np.ndarray):
    """Return stats and label map for connected components sorted by area desc (skip background)."""
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask01.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x, y, w, h, a = int(stats[i, 0]), int(stats[i, 1]), int(
            stats[i, 2]), int(stats[i, 3]), area
        comps.append({"label": i, "area": a, "bbox": (x, y, w, h)})
    comps.sort(key=lambda d: d["area"], reverse=True)
    return comps, labels


def initialize_model():
    global model, sam, predictor, device, transform

    model = models.efficientnet_v2_s(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, No_classes)

    checkpoint = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def sample_points_within_contour(contour, num_points):
    rect = cv2.boundingRect(contour)
    mask = np.zeros((rect[3], rect[2]), dtype=np.uint8)
    shifted_contour = contour - np.array([[rect[0], rect[1]]])
    cv2.drawContours(mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
    ys, xs = np.where(mask == 255)
    if len(xs) < num_points:
        return [(xs[i] + rect[0], ys[i] + rect[1]) for i in range(len(xs))]
    sampled_indices = random.sample(range(len(xs)), num_points)
    return [(xs[i] + rect[0], ys[i] + rect[1]) for i in sampled_indices]


# --- Thresholding and Segmentation ---
def process_images_in_batch(image_paths, target_class, threshold_value, num_sampled_points, save_folder):
    for image_path in image_paths:
        try:
            # Load and process the image
            original_image = Image.open(image_path).convert('RGB')
            input_tensor = transform(original_image).unsqueeze(0).to(device)
            
            # Generate GradCAM
            cam = GradCAM(model=model, target_layers=[model.features[-1]])
            grayscale_cams = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_class)])

            # Resize and apply thresholding
            grayscale_cam = grayscale_cams[0]
            grayscale_cam_resized = cv2.resize(grayscale_cam, original_image.size, interpolation=cv2.INTER_LINEAR)
            _, binary_map = cv2.threshold(np.uint8(255 * grayscale_cam_resized), threshold_value, 255, cv2.THRESH_BINARY)

            # Find contours in the binary map
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sample points and generate segmentation mask using SAM
            all_sampled_points, all_input_labels = [], []
            for contour in contours:
                sampled_points = sample_points_within_contour(contour, num_sampled_points)
                all_sampled_points.extend(sampled_points)
                all_input_labels.extend([1] * len(sampled_points))

            if all_sampled_points:
                predictor.set_image(np.array(original_image))
                masks, scores, logits = predictor.predict(
                    point_coords=np.array(all_sampled_points),
                    point_labels=np.array(all_input_labels, dtype=np.int32),
                    multimask_output=True
                )

                best_mask_index = np.argmax(scores)
                best_mask_input = logits[best_mask_index, :, :]

                # Refine the mask using SAM
                refined_mask, _, _ = predictor.predict(
                    point_coords=np.array(all_sampled_points),
                    point_labels=np.array(all_input_labels, dtype=np.int32),
                    mask_input=best_mask_input[None, :, :],
                    multimask_output=False
                )

                # Post-process to clean up small components and holes
                refined_mask = np.squeeze(refined_mask).astype(np.uint8)

                # 0/1 → clean up → 0/1
                bin_mask = (refined_mask > 0).astype(np.uint8)
                bin_mask = _remove_small_components(bin_mask, SPECKLE_REMOVE_MAX_FRAC)
                bin_mask = _fill_small_holes(bin_mask, HOLE_FILL_MAX_FRAC)

                # 0/1 → 0/255 for true binary mask
                bin_mask = (bin_mask * 255).astype(np.uint8)    
                # Always save as PNG to avoid JPEG artifacts
                base = Path(image_path).stem          # e.g. "obs_12345_photo_6789"
                mask_save_path = os.path.join(save_folder, f"mask_{base}.png")

                cv2.imwrite(mask_save_path, bin_mask)
                print(f"Saved refined BINARY mask to {mask_save_path}")

            else:
            # Save empty all-background mask if no valid contours were found
                h, w = original_image.size[1], original_image.size[0]
                empty_mask = np.zeros((h, w), dtype=np.uint8)  # all 0 = background
                base = Path(image_path).stem
                mask_save_path = os.path.join(save_folder, f"mask_{base}.png")
                cv2.imwrite(mask_save_path, empty_mask)
                print(f"Saved EMPTY mask for {image_path}") 
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")


def get_class_image_paths_and_savefolders(base_dir):
    """
    Returns a list of (class_name, image_paths, save_folder) where:
    - Each top-level folder is a class.
    - For folders with subfolders (like '001_conifers'), all images in all subfolders are grouped as one class,
      and masks are saved in the corresponding subfolder under <main_folder>_mask.
    """
    class_image_paths = []
    for folder in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        save_root = os.path.join(base_dir, f"{folder}_mask")
        if folder.lower() == "001_conifers":
            # Recursively walk through all subfolders
            for root, dirs, files in os.walk(folder_path):
                rel_root = os.path.relpath(root, folder_path)
                save_folder = os.path.join(
                    save_root, rel_root) if rel_root != "." else save_root
                os.makedirs(save_folder, exist_ok=True)
                image_paths = [os.path.join(
                    root, fname) for fname in files if fname.lower().endswith(patterns)]
                if image_paths:
                    class_image_paths.append(
                        ("001_conifers", image_paths, save_folder))
        else:
            # Regular class: all images in this folder and subfolders
            for root, dirs, files in os.walk(folder_path):
                rel_root = os.path.relpath(root, folder_path)
                save_folder = os.path.join(
                    save_root, rel_root) if rel_root != "." else save_root
                os.makedirs(save_folder, exist_ok=True)
                image_paths = [os.path.join(
                    root, fname) for fname in files if fname.lower().endswith(patterns)]
                if image_paths:
                    class_image_paths.append(
                        (folder, image_paths, save_folder))
    return class_image_paths


def process_all_folders():
    class_image_paths = get_class_image_paths_and_savefolders(base_dir)
    # Build class_name to class_idx mapping (order matters!)
    class_names = []
    for class_name, _, _ in class_image_paths:
        if class_name not in class_names:
            class_names.append(class_name)
    class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for class_name, image_paths, save_folder in class_image_paths:
        class_idx = class_name_to_idx[class_name]
        print(
            f"\n🟩 Class: {class_name} | Index: {class_idx} | {len(image_paths)} images")
        initialize_model()
        for i in range(0, len(image_paths), Batch_size):
            batch_paths = image_paths[i:i + Batch_size]
            process_images_in_batch(
                batch_paths, class_idx, Threshold_value, No_of_sampled_points, save_folder)


if __name__ == "__main__":
    try:
        process_all_folders()
    except Exception as e:
        logger.error(f"Error in main process: {e}")
