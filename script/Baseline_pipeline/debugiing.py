import os

image_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
mask_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Masks"

# Debugging: print out the files found in both directories
print("Images in directory:", os.listdir(image_dir))
print("Masks in directory:", os.listdir(mask_dir))

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

pairs = []
for img_name in image_files:
    base, _ = os.path.splitext(img_name)
    # Try matching multiple extensions for masks
    for ext in [".png", ".jpg", ".jpeg"]:
        mask_path = os.path.join(mask_dir, base + ext)
        if os.path.exists(mask_path):
            pairs.append((os.path.join(image_dir, img_name), mask_path))
            break

print(f"Found {len(pairs)} image-mask pairs.")

# If no pairs found, raise an error
if not pairs:
    raise RuntimeError(
        "No image–mask pairs found. Check folder paths and names."
    )
