# mask_preview_fixed_images_only.py
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random

# ========================
# USER CONFIGURATION
# ========================
IMAGES_ROOT = r"E:/Santosh_master_thesis/LOT_all_images_labeled"
MASKS_ROOT = r"E:/Santosh_master_thesis/LOT_masks_labels"
MASK_CLASS_SUFFIX = "_mask"
MASK_PREFIX = "mask_"
BACKGROUND_VALUE = 255
OUT_PATH = r"E:/Santosh_master_thesis/LOT_all_images_labeled/plot_image_masks.png"

NUM_EXAMPLES = 4
MAX_SUBPLOTS = 160
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# ========================
# COLORS
# ========================
bg_color = "#86BFBA"
fg_color = "#0A55EC"
cmap = mcolors.ListedColormap([bg_color, fg_color])

# ========================
# SCRIPT
# ========================
mask_classes = [c for c in os.listdir(MASKS_ROOT) if c.endswith(MASK_CLASS_SUFFIX)]
mask_classes.sort()

num_rows = min(len(mask_classes), MAX_SUBPLOTS // 2)
num_cols_adjusted = min(2 * NUM_EXAMPLES + 1, MAX_SUBPLOTS + num_rows)

plt.figure(figsize=(20, 20))
plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98,
                    top=0.98, wspace=0.1, hspace=0.1)

for i, mask_class in enumerate(mask_classes):
    mask_folder = os.path.join(MASKS_ROOT, mask_class)
    mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(IMG_EXTS)]
    masks = random.sample(mask_files, min(NUM_EXAMPLES, len(mask_files)))

    # add class name as row label
    plt.figtext(0.01, 1 - (i + 0.5) / num_rows,
                f'{mask_class[:-len(MASK_CLASS_SUFFIX)]}',
                ha='left', va='center', fontsize=12, fontweight='bold')

    for j, mask_name in enumerate(masks):
        mask_path = os.path.join(mask_folder, mask_name)

        # corresponding image file (strip mask prefix + match extension)
        base_name = mask_name[len(MASK_PREFIX):-4]  # remove prefix + extension
        found = False
        for ext in IMG_EXTS:
            image_path = os.path.join(IMAGES_ROOT,
                                      mask_class[:-len(MASK_CLASS_SUFFIX)],
                                      base_name + ext)
            if os.path.exists(image_path):
                found = True
                break
        if not found:
            print(f"Image for {mask_name} not found.")
            continue

        # load image + mask
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error loading {image_path} or {mask_path}")
            continue

        # binarize mask
        mask_binary = np.where(mask == BACKGROUND_VALUE, 0, 1)

        image_subplot_index = i * num_cols_adjusted + 2 * j + 2
        mask_subplot_index = i * num_cols_adjusted + 2 * j + 3

        plt.subplot(num_rows, num_cols_adjusted, image_subplot_index)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        plt.subplot(num_rows, num_cols_adjusted, mask_subplot_index)
        plt.imshow(mask_binary, cmap=cmap, vmin=0, vmax=1)
        plt.axis('off')

# save and show
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH)
plt.show()

print(f"✅ Saved preview to {OUT_PATH}")
