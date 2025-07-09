import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random

# Settings
BACKGROUND_VALUE = 10
bg_color = "#86BFBA"
fg_color = "#0A55EC"
cmap = mcolors.ListedColormap([bg_color, fg_color])
data_folder = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/iNaturalist"
num_examples = 4  # Number of images per subfolder to plot
max_subplots = 160  # Maximum number of subplots per figure

# Get all main class folders (exclude _mask folders)
main_folders = [f for f in os.listdir(data_folder)
                if os.path.isdir(os.path.join(data_folder, f)) and not f.endswith("_mask")]

# ...existing code...

for main_class in sorted(main_folders):
    image_main_folder = os.path.join(data_folder, main_class)
    mask_main_folder = os.path.join(data_folder, main_class + "_mask")
    if not os.path.exists(mask_main_folder):
        print(f"Mask folder not found for {main_class}")
        continue

    # Check for subfolders
    subfolders = [f for f in os.listdir(image_main_folder)
                  if os.path.isdir(os.path.join(image_main_folder, f))]

    # If no subfolders, treat the main folder as a single subfolder
    if not subfolders:
        subfolders = [""]
        print(f"No subfolders found in {image_main_folder}. Treating as single folder.")

    num_rows = len(subfolders)
    num_cols = 2 * num_examples  # Each example: image + mask

    plt.figure(figsize=(4 * num_cols, 4 * num_rows))
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98, wspace=0.1, hspace=0.2)

    for i, subfolder in enumerate(sorted(subfolders)):
        if subfolder == "":
            image_folder = image_main_folder
            mask_folder = mask_main_folder
            subfolder_name = main_class
        else:
            image_folder = os.path.join(image_main_folder, subfolder)
            mask_folder = os.path.join(mask_main_folder, subfolder + "_mask")
            subfolder_name = subfolder

        if not os.path.exists(mask_folder):
            print(f"Mask folder not found for {subfolder_name} in {main_class}")
            continue

        masks = [f for f in os.listdir(mask_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not masks:
            print(f"No mask files found in {mask_folder}")
            continue
        masks = random.sample(masks, min(num_examples, len(masks)))

        for j, mask_name in enumerate(masks):
            mask_path = os.path.join(mask_folder, mask_name)

            # Try all possible image extensions
            found = False
            for ext in [".jpg", ".png", ".jpeg"]:
                if mask_name.startswith("mask_"):
                    image_base = mask_name[5:-4]
                else:
                    image_base = mask_name[:-4]
                image_name = image_base + ext
                image_path = os.path.join(image_folder, image_name)
                if os.path.exists(image_path):
                    found = True
                    break
            if not found:
                print(f"Image file for {mask_name} not found with any extension in {image_folder}.")
                continue

            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Error loading image or mask: {image_path} / {mask_path}")
                continue

            mask_binary = np.where(mask == BACKGROUND_VALUE, 0, 1)

            # Plot image
            plt.subplot(num_rows, num_cols, i * num_cols + 2 * j + 1)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f"{subfolder_name} - Img", fontsize=10)
            plt.axis('off')

            # Plot mask
            plt.subplot(num_rows, num_cols, i * num_cols + 2 * j + 2)
            plt.imshow(mask_binary, cmap=cmap, vmin=0, vmax=1)
            plt.title(f"{subfolder_name} - Mask", fontsize=10)
            plt.axis('off')

    save_path = os.path.join(data_folder, f"plot_{main_class}_image_masks.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved plot for {main_class} at {save_path}")
# End of script
    