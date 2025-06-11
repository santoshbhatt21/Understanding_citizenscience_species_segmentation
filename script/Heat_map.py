import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import random
import torch.nn as nn
from collections import OrderedDict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup base directory and parameters
base_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/iNaturalist"
Threshold_value = 80
No_of_sampled_points = 2
No_classes = 7
Batch_size = 10  # Adjust batch size based on your GPU capacity
Background_class = 120 
Contour_line_size = 2
Point_radius = 5
Sampled_images_per_folder = 10
contour_color = (255, 255, 255)  # White contour
contour_thickness = 7
point_color = (0, 0, 0)   # White points
point_radius = 10

# Load models and preprocessing
model_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point/best_model_39_0.21.pth"
patterns = tuple(['.jpg', '.png', '.JPEG', '.JPG', '.PNG', '.jpeg'])

def initialize_model():
    global model, device, transform

    # Initialize the EfficientNet model
    model = models.efficientnet_v2_l(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, No_classes)

    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        # Remove 'module.' prefix if it exists (for compatibility with DataParallel models)
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=False)

    # Set device to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

def process_images_in_batch(image_paths, target_class, Threshold_value, No_of_sampled_points, save_folder):
    try:
        global model, transform, device

        batch_images = []
        original_images = []
        for image_path in image_paths:
            original_image = Image.open(image_path).convert('RGB')
            original_images.append((image_path, original_image))
            input_tensor = transform(original_image).unsqueeze(0)
            batch_images.append(input_tensor)

        batch_input_tensor = torch.cat(batch_images).to(device)

        # Initialize GradCAM with the last layer of the feature extractor
        cam = GradCAM(model=model, target_layers=[model.features[-1]])
        grayscale_cams = cam(input_tensor=batch_input_tensor, targets=[ClassifierOutputTarget(target_class)] * len(image_paths))

        for idx, (image_path, original_image) in enumerate(original_images):
            grayscale_cam = grayscale_cams[idx]
            grayscale_cam_resized = cv2.resize(grayscale_cam, original_image.size, interpolation=cv2.INTER_LINEAR)
            heatmap = np.uint8(255 * grayscale_cam_resized)

            # Apply colormap and invert the heatmap colors
            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            heatmap_color = cv2.bitwise_not(heatmap_color)  # Invert the heatmap colors

            # Overlay heatmap on the original image
            original_image_np = np.array(original_image) / 255.0
            heatmap_overlay = cv2.addWeighted(original_image_np, 0.6, heatmap_color / 255.0, 0.4, 0)

            # Save the heatmap-overlayed image without contours or points
            heatmap_no_contour_save_path = os.path.join(save_folder, f'heatmap_overlay_{os.path.basename(image_path)}')
            heatmap_no_contour_image = np.uint8(255 * heatmap_overlay)
            Image.fromarray(heatmap_no_contour_image).save(heatmap_no_contour_save_path)


            # Ensure proper format for heatmap_overlay
            heatmap_overlay = np.uint8(255 * heatmap_overlay) if heatmap_overlay.max() <= 1.0 else np.uint8(heatmap_overlay)

            # Draw contours and points on the heatmap-overlayed image
            _, binary_map = cv2.threshold(heatmap, Threshold_value, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Draw contours
                # Set desired thickness for the contour
                cv2.drawContours(heatmap_overlay, [contour], -1, contour_color, contour_thickness)

                # Sample and draw points within the contour
                sampled_points = sample_points_within_contour(contour, No_of_sampled_points)
              # Set desired radius for the points
                for point in sampled_points:
                    cv2.circle(heatmap_overlay, point, point_radius, point_color, -1)  # Filled circle

            # Save the overlayed visualization with contours and points
            visualization = np.uint8(heatmap_overlay)
            vis_image = Image.fromarray(visualization)
            vis_save_path = os.path.join(save_folder, f'vis_{os.path.basename(image_path)}')
            vis_image.save(vis_save_path)
            logger.info(f"Visualization saved to {vis_save_path}")

    except Exception as e:
        logger.error(f"Error processing images in batch: {e}")
    finally:
        torch.cuda.empty_cache()

def process_folder(subdir, folder_idx, Threshold_value, No_of_sampled_points):
    initialize_model()
    target_class = folder_idx
    subdir_path = os.path.join(base_dir, subdir)
    if os.path.isdir(subdir_path):
        save_folder = f'{subdir_path}_heatmap'
        os.makedirs(save_folder, exist_ok=True)
        image_paths = [os.path.join(subdir_path, image_name) for image_name in os.listdir(subdir_path) if os.path.isfile(os.path.join(subdir_path, image_name)) and image_name.lower().endswith(patterns)]

        # Sample 100 images from the folder
        sampled_image_paths = random.sample(image_paths, min(Sampled_images_per_folder, len(image_paths)))

        for i in range(0, len(sampled_image_paths), Batch_size):
            batch_paths = sampled_image_paths[i:i + Batch_size]
            process_images_in_batch(batch_paths, target_class, Threshold_value, No_of_sampled_points, save_folder)

if __name__ == "__main__":
    try:
        for folder_idx, subdir in enumerate(sorted(os.listdir(base_dir))):
            process_folder(subdir, folder_idx, Threshold_value, No_of_sampled_points)
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        logger.info("Processing completed.")
        torch.cuda.empty_cache()