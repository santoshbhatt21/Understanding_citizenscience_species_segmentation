import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

val_image_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/YOLO/images/val"
output_path = "val_batch2_pred.jpg"

# Recursively collect all image file paths from all subfolders
image_paths = []
for root, dirs, files in os.walk(val_image_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(os.path.join(root, file))

print(f"Found {len(image_paths)} images for validation.")

if len(image_paths) == 0:
    raise ValueError("No images found in the validation directory or its subfolders.")

# Load a batch of images (e.g., first 16)
image_size = (256, 256)
batch_paths = image_paths[:16]
batch = []
for img_path in batch_paths:
    img = Image.open(img_path).convert("RGB").resize(image_size)
    batch.append(np.array(img))

if len(batch) == 0:
    raise ValueError("No images loaded for visualization.")

# Visualization: plot images in a grid
ncols = 4
nrows = int(np.ceil(len(batch) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*4))

for idx, ax in enumerate(axes.flat):
    if idx < len(batch):
        ax.imshow(batch[idx])
        ax.axis('off')
        ax.set_title(os.path.basename(batch_paths[idx]))
    else:
        ax.axis('off')

plt.tight_layout()
plt.savefig(output_path)
plt.close()
print(f"Saved visualization to {output_path}")