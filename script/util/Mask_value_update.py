import os
import numpy as np
import cv2
from tqdm import tqdm
import multiprocessing
from functools import partial

def process_mask(mask_path):
    # Read the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    
    # Check if the mask was loaded successfully
    if mask is None:
        print(f"Failed to load image: {mask_path}")
        return
    
    # Change the mask value where it is 120 to 10
    mask = mask+1 # Add 1 to all values
    
    # Save the modified mask back to the same location
    cv2.imwrite(mask_path, mask)

def process_masks_in_folder(folder_path, num_cpus):
    # List all mask files in the folder and subfolders
    mask_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".png"):  # Adjust the file extension if needed
                mask_path = os.path.join(root, file)
                mask_files.append(mask_path)
    
    # Use multiprocessing to process the mask files
    with multiprocessing.Pool(processes=num_cpus) as pool:
        list(tqdm(pool.imap_unordered(process_mask, mask_files), total=len(mask_files), desc="Processing masks", unit="file"))

# Path to the main folder containing the three folders with masks
main_folder_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/iNaturalist"

# Number of CPUs to use
num_cpus = 10 # You can adjust this to the desired number of CPUs

# Process each folder
process_masks_in_folder(main_folder_path, num_cpus)
