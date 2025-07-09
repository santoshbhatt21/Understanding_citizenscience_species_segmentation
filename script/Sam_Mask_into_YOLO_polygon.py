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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup base directory and parameters
base_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/iNaturalist"

Threshold_value = 80  # Medium focused with details
No_of_sampled_points = 2
No_classes = 6  # 5 folders + 1 conifers (with all subfolders as one class)
Batch_size = 32
Background_class = 10

model_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/checkpoints/best_model_47_0.22.pth"
sam_checkpoint = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"

patterns = tuple(['.jpg', '.png', '.JPEG', '.JPG', '.PNG', '.jpeg'])

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

# --- YOLO Polygon Conversion Function ---
def mask_to_yolo_polygon(mask_path, class_id, save_txt_path):
    """
    Converts a mask image to YOLO polygon format and saves as a .txt file.
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask: {mask_path}")
        return

    # Only keep the class_id region as foreground
    binary_mask = (mask == class_id).astype(np.uint8)
    if np.sum(binary_mask) == 0:
        return

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape

    with open(save_txt_path, 'w') as f:
        for contour in contours:
            if len(contour) < 3:
                continue  # Not a valid polygon
            polygon = contour.squeeze().astype(float)
            if polygon.ndim == 1:
                polygon = polygon.reshape(-1, 2)
            polygon[:, 0] /= w  # x / width
            polygon[:, 1] /= h  # y / height
            coords = polygon.flatten().tolist()
            coords_str = ' '.join([f"{c:.6f}" for c in coords])
            f.write(f"{class_id} {coords_str}\n")
# --- END YOLO Polygon Conversion Function ---

def process_images_in_batch(image_paths, target_class, threshold_value, num_sampled_points, save_folder):
    try:
        global model, predictor, transform, device

        batch_images = []
        original_images = []
        for image_path in image_paths:
            original_image = Image.open(image_path).convert('RGB')
            original_images.append((image_path, original_image))
            input_tensor = transform(original_image).unsqueeze(0)
            batch_images.append(input_tensor)
        batch_input_tensor = torch.cat(batch_images).to(device)

        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        grayscale_cams = cam(input_tensor=batch_input_tensor,
                             targets=[ClassifierOutputTarget(target_class)] * len(image_paths))

        for idx, (image_path, original_image) in enumerate(original_images):
            grayscale_cam = grayscale_cams[idx]
            grayscale_cam_resized = cv2.resize(
                grayscale_cam, original_image.size, interpolation=cv2.INTER_LINEAR)
            _, binary_map = cv2.threshold(
                np.uint8(255 * grayscale_cam_resized), threshold_value, 255, cv2.THRESH_BINARY)

            contours, _ = cv2.findContours(
                binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            all_sampled_points, all_input_labels = [], []
            for contour in contours:
                sampled_points = sample_points_within_contour(
                    contour, num_sampled_points)
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

                refined_mask, _, _ = predictor.predict(
                    point_coords=np.array(all_sampled_points),
                    point_labels=np.array(all_input_labels, dtype=np.int32),
                    mask_input=best_mask_input[None, :, :],
                    multimask_output=False
                )
                refined_mask = np.squeeze(refined_mask)
                final_mask = np.where(
                    refined_mask, target_class, Background_class).astype(np.uint8)
                mask_save_path = os.path.join(
                    save_folder, f'mask_{os.path.splitext(os.path.basename(image_path))[0]}.png')
                cv2.imwrite(mask_save_path, final_mask)
                logger.info(f"Refined mask saved to {mask_save_path}")

                # --- YOLO Polygon Export ---
                yolo_txt_path = mask_save_path.replace('.png', '.txt')
                mask_to_yolo_polygon(mask_save_path, target_class, yolo_txt_path)
                logger.info(f"YOLO polygon annotation saved to {yolo_txt_path}")
                # --- END YOLO Polygon Export ---

            else:
                logger.info(
                    f"No activation contours found for {image_path}; skipping mask generation.")
    except Exception as e:
        logger.error(f"Error processing images in batch: {e}")
    finally:
        torch.cuda.empty_cache()

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
                save_folder = os.path.join(save_root, rel_root) if rel_root != "." else save_root
                os.makedirs(save_folder, exist_ok=True)
                image_paths = [os.path.join(root, fname) for fname in files if fname.lower().endswith(patterns)]
                if image_paths:
                    class_image_paths.append(("001_conifers", image_paths, save_folder))
        else:
            # Regular class: all images in this folder and subfolders
            for root, dirs, files in os.walk(folder_path):
                rel_root = os.path.relpath(root, folder_path)
                save_folder = os.path.join(save_root, rel_root) if rel_root != "." else save_root
                os.makedirs(save_folder, exist_ok=True)
                image_paths = [os.path.join(root, fname) for fname in files if fname.lower().endswith(patterns)]
                if image_paths:
                    class_image_paths.append((folder, image_paths, save_folder))
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
        print(f"\n🟩 Class: {class_name} | Index: {class_idx} | {len(image_paths)} images")
        initialize_model()
        for i in range(0, len(image_paths), Batch_size):
            batch_paths = image_paths[i:i + Batch_size]
            process_images_in_batch(batch_paths, class_idx, Threshold_value, No_of_sampled_points, save_folder)

if __name__ == "__main__":
    try:
        process_all_folders()
    except Exception as e:
        logger.error(f"Error in main process: {e}")