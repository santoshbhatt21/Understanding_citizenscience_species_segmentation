import numpy as np
import cv2
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segment_anything import SamPredictor

# Update these parameters for dynamic and adaptive thresholding
Threshold_value = 50  # Adjusted threshold for better segmentation
SPECKLE_REMOVE_MAX_FRAC = 0.01  # Slightly less aggressive speckle removal (up to 1% of image area)
HOLE_FILL_MAX_FRAC = 0.005  # Slightly less aggressive hole filling (up to 0.5% of image area)

# --- Post-Processing functions (unchanged) ---
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
                bin_mask = (refined_mask > 0).astype(np.uint8)
                bin_mask = _remove_small_components(bin_mask, SPECKLE_REMOVE_MAX_FRAC)
                bin_mask = _fill_small_holes(bin_mask, HOLE_FILL_MAX_FRAC)

                # Save final refined binary mask
                mask_save_path = os.path.join(save_folder, f"mask_{os.path.basename(image_path)}")
                cv2.imwrite(mask_save_path, bin_mask)
                print(f"Saved refined mask for {image_path}")

            else:
                # Save empty mask if no valid contours were found
                empty_mask = np.full(refined_mask.shape, Background_class, dtype=np.uint8)
                cv2.imwrite(os.path.join(save_folder, f"mask_{os.path.basename(image_path)}"), empty_mask)
                print(f"Saved empty mask for {image_path}")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

# Example of how to adjust Threshold and Post-Processing
Threshold_value = 60  # Change this threshold based on the intensity of CAM outputs
SPECKLE_REMOVE_MAX_FRAC = 0.01  # Less aggressive removal of small components (up to 1% of image area)
HOLE_FILL_MAX_FRAC = 0.005  # Less aggressive hole-filling (up to 0.5% of image area)
