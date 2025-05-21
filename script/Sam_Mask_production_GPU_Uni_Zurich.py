
# Add the functionality to your code that can read files recursively
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
base_dir = 'E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data'

Threshold_value = 80  # 150
No_of_sampled_points = 2
No_classes = 6
Batch_size = 16  # Adjust batch size based on your GPU capacity
Background_class = 10

# Load models and preprocessing
model_path = '/mnt/gsdata/projects/bigplantsens/5_ETH_Zurich_Citizen_Science_Segment/Checkpoint/best_model_68_0.02.pth'
sam_checkpoint = '/mnt/gsdata/projects/bigplantsens/1_FloraMask/4_F_Japonica/checkpoint/F_japonica_cam/sam_vit_h_4b8939.pth'

patterns = tuple(['.jpg', '.png', '.JPEG', '.JPG', '.PNG', '.jpeg'])


def initialize_model():
    global model, sam, predictor, device, transform

    model = models.efficientnet_v2_l(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, No_classes)

    checkpoint = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
    sam.to(device)
    predictor = SamPredictor(sam)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
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


def process_images_in_batch(image_paths, target_class, threshold_value, num_sampled_points, save_folder):
    try:
        global model, predictor, transform, device

        # STEP 1: Preprocessing & Activation-Based Prompting
        batch_images = []
        original_images = []
        for image_path in image_paths:
            original_image = Image.open(image_path).convert('RGB')
            original_images.append((image_path, original_image))
            input_tensor = transform(original_image).unsqueeze(0)
            batch_images.append(input_tensor)
        batch_input_tensor = torch.cat(batch_images).to(device)

        # Generate activation maps using GradCAM for the given target class
        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        grayscale_cams = cam(input_tensor=batch_input_tensor,
                             targets=[ClassifierOutputTarget(target_class)] * len(image_paths))

        # Process each image individually
        for idx, (image_path, original_image) in enumerate(original_images):
            # Resize activation map to original image size and threshold it
            grayscale_cam = grayscale_cams[idx]
            grayscale_cam_resized = cv2.resize(
                grayscale_cam, original_image.size, interpolation=cv2.INTER_LINEAR)
            _, binary_map = cv2.threshold(
                np.uint8(255 * grayscale_cam_resized), threshold_value, 255, cv2.THRESH_BINARY)

            # Find contours on the binary map to extract candidate regions
            contours, _ = cv2.findContours(
                binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sample points from the activated regions (using your helper function)
            all_sampled_points, all_input_labels = [], []
            for contour in contours:
                sampled_points = sample_points_within_contour(
                    contour, num_sampled_points)
                all_sampled_points.extend(sampled_points)
                all_input_labels.extend([1] * len(sampled_points))

            if all_sampled_points:
                # STEP 2: Multimask Prediction with SAM
                predictor.set_image(np.array(original_image))
                masks, scores, logits = predictor.predict(
                    point_coords=np.array(all_sampled_points),
                    point_labels=np.array(all_input_labels, dtype=np.int32),
                    multimask_output=True  # Generate several candidate masks
                )

                # STEP 3: Score-Based Selection & Refinement
                # Select the mask with the highest score as the candidate for refinement
                best_mask_index = np.argmax(scores)
                best_mask_input = logits[best_mask_index, :, :]

                # Refine the mask by feeding it back into SAM using the best mask as an input
                refined_mask, _, _ = predictor.predict(
                    point_coords=np.array(all_sampled_points),
                    point_labels=np.array(all_input_labels, dtype=np.int32),
                    mask_input=best_mask_input[None, :, :],
                    multimask_output=False  # Only one refined output is needed
                )
                refined_mask = np.squeeze(refined_mask)

                # Create final mask: assign target_class for the foreground and Background_class for background
                final_mask = np.where(
                    refined_mask, target_class, Background_class).astype(np.uint8)
                mask_save_path = os.path.join(
                    save_folder, f'mask_{os.path.splitext(os.path.basename(image_path))[0]}.png')
                cv2.imwrite(mask_save_path, final_mask)
                logger.info(f"Refined mask saved to {mask_save_path}")
            else:
                logger.info(
                    f"No activation contours found for {image_path}; skipping mask generation.")
    except Exception as e:
        logger.error(f"Error processing images in batch: {e}")
    finally:
        torch.cuda.empty_cache()


def process_folder(subdir, folder_idx, Threshold_value, No_of_sampled_points):
    initialize_model()
    target_class = folder_idx
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        save_folder = os.path.join(root, f'{subdir}_mask')
        os.makedirs(save_folder, exist_ok=True)
        image_paths = [os.path.join(subdir_path, image_name) for image_name in os.listdir(subdir_path) if os.path.isfile(
            os.path.join(subdir_path, image_name)) and image_name.lower().endswith(patterns)]

        for i in range(0, len(image_paths), Batch_size):
            batch_paths = image_paths[i:i + Batch_size]
            process_images_in_batch(
                batch_paths, target_class, Threshold_value, No_of_sampled_points, save_folder)


if __name__ == "__main__":
    try:
        for root, dirs, files in os.walk(base_dir):
            for folder_idx, subdir in enumerate(sorted(dirs)):
                process_folder(subdir, folder_idx,
                               Threshold_value, No_of_sampled_points)

    except Exception as e:
        logger.error(f"Error in main process: {e}")
