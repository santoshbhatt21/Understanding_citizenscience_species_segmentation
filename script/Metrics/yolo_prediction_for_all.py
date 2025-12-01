import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import numpy as np

# =========================
# CONFIG – EDIT THIS PART
# =========================

# Root folders for MASKS (grayscale masks: 0 = non-target, 1 = target, etc.)
MASK_ROOT_BASELINE = r"E:/Santosh_master_thesis/Classified_Masks_binary"
OUTPUT_DIR = r"E:/Santosh_master_thesis/Yolo_Predictions_Output"

# Best YOLO model path
MODEL_PATH = r"E:/Santosh_master_thesis/species_segmentation_leaves/yolo11_leaves_seg_final/weights/best.pt"

# Confidence threshold for valid predictions (you can adjust this value)
CONF_THRESHOLD = 0.25

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# =========================
# HELPER FUNCTIONS
# =========================


def list_images(root, exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff")):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in exts]
    return files


def save_yolo_predictions(results, output_path, base_name: str):
    """Save YOLO predictions for a single image into its own folder.

    output_path: base folder where species-level subfolders live
    base_name:   stem of the mask image (used to name the subfolder)
    """
    output_path = Path(output_path) / base_name
    output_path.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    for i, result in enumerate(results):
        prediction_image = result.plot()
        prediction_image = Image.fromarray(prediction_image)
        out_file = output_path / f"prediction_{i}.png"
        prediction_image.save(out_file)
        print(f"Saved prediction to {out_file}")


def run_yolo_predictions(model, image_path, conf_threshold=CONF_THRESHOLD):
    """Run YOLO prediction for a given image and return non-empty results."""
    # Let YOLO handle the confidence threshold internally
    results = model(image_path, conf=conf_threshold)

    # Keep only results that actually have detections
    valid_results = [r for r in results if (
        r.boxes is not None and len(r.boxes) > 0)]
    return valid_results

# =========================
# MAIN
# =========================


if __name__ == "__main__":
    # Load the best YOLO model
    # Ensure you provide the correct path to your best YOLO model
    model = YOLO(MODEL_PATH)

    # Loop over all mask folders
    for species_folder in Path(MASK_ROOT_BASELINE).iterdir():
        if not species_folder.is_dir():
            continue

        print(f"Processing folder: {species_folder.name}")

        # List all images in the folder
        image_files = list_images(species_folder)

        # Loop over each image in the folder
        for img_path in image_files:
            # Run YOLO prediction for each image (mask)
            print(f"Running YOLO on {img_path.name}...")
            valid_predictions = run_yolo_predictions(model, img_path)

            # If there are valid predictions, save them in
            # OUTPUT_DIR/<species_folder>/<mask_stem>/prediction_*.png
            if valid_predictions:
                species_out_dir = Path(OUTPUT_DIR) / species_folder.name
                save_yolo_predictions(
                    valid_predictions, species_out_dir, img_path.stem)
            else:
                print(f"No valid predictions for {img_path.name}")

    print("YOLO predictions completed for all masks.")
