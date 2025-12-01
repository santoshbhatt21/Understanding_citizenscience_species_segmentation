import cv2
import numpy as np
# Load the mask in grayscale mode (single channel)
mask_path = "E:/Santosh_master_thesis/Classified_Leaves/Betula_pendula_Leaves_mask/mask_obs_224182670_photo_397291952.png"  # Replace with your mask file path
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

# Check the unique values in the mask to confirm it's binary
unique_values = np.unique(mask)
print("Unique values in the mask:", unique_values)
