import os

root_dir = r"E:/Santosh_master_thesis/Generated_YOLO_Seg_Labels_Leaves_inverted"
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


overall_total = 0

for species in sorted(os.listdir(root_dir)):
    species_path = os.path.join(root_dir, species)
    if not os.path.isdir(species_path):
        continue

    image_count = sum(
        1 for fname in os.listdir(species_path)
        if fname.lower().endswith(image_extensions)
    )
    print(f"{species}: {image_count} images")
    overall_total += image_count

print(f"\n🌟 Total images in all species folders: {overall_total}")