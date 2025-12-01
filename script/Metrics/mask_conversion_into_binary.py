import numpy as np
import cv2
from pathlib import Path

# Set the input and output directories
root_in = Path(r"E:/Santosh_master_thesis/Classified_Leaves_Masks")
root_out = Path(r"E:/Santosh_master_thesis/Classified_Leaves_Masks_binary")
root_out.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist

# Define valid image extensions
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

def convert_to_binary_mask(mask_path, output_path):
    """
    Converts a multi-channel mask image to a binary mask (black and white) and saves it to the output path.
    """
    # Load the multi-channel mask image
    mask = cv2.imread(mask_path)

    # Ensure the mask is not empty
    if mask is None:
        print(f"Error: Could not load image {mask_path}")
        return

    # Convert to grayscale (if the mask is multi-channel, this will collapse it to a single channel)
    grayscale_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to get a binary mask: all non-zero pixels become 255 (foreground), others become 0 (background)
    _, binary_mask = cv2.threshold(grayscale_mask, 1, 255, cv2.THRESH_BINARY)

    # Save the binary mask in the output directory, maintaining the directory structure
    cv2.imwrite(output_path, binary_mask)
    print(f"Binary mask saved at {output_path}")

# Process all mask images in the input directory and save the binary masks
for path in root_in.rglob("*"):
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        # Get relative path to maintain directory structure
        rel = path.relative_to(root_in)
        out_path = root_out / rel  # Output path where the binary mask will be saved
        out_path.parent.mkdir(parents=True, exist_ok=True)  # Create output subdirectories if needed

        # Convert the mask to binary and save it
        convert_to_binary_mask(str(path), str(out_path))
print("All masks have been converted to binary format.")