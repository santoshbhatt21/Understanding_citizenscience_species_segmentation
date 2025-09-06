import os

# === Root path to your sorted dataset ===
root_dir = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labelling_Manual" #"E:/Santosh_master_thesis/DATA_YOLO11_LT_clean_labels/labels"

# === Define image extensions to count ===
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
text_extensions = ('.txt',)

# === Count images in all subfolders ===
total_images = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    count = sum(
        1 for fname in filenames if fname.lower().endswith(image_extensions)
    )
    if count > 0:
        print(f"{dirpath}: {count} images")
    total_images += count

print(f"\n🌟 Total images in all subfolders: {total_images}")

# === Count label .txt files in all subfolders (same style) ===
total_labels = 0
for dirpath, dirnames, filenames in os.walk(root_dir):
    count = sum(
        1 for fname in filenames if fname.lower().endswith(text_extensions)
    )
    if count > 0:
        print(f"{dirpath}: {count} labels")
    total_labels += count

print(f"\n📝 Total labels (.txt) in all subfolders: {total_labels}")

overall_total_images = 0
overall_total_labels = 0

# === Traverse classes and species subfolders ===
for class_name in sorted(os.listdir(root_dir)):
    class_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📂 Class: {class_name}")
    class_total_images = 0
    class_total_labels = 0
    for species in sorted(os.listdir(class_path)):
        species_path = os.path.join(class_path, species)
        if not os.path.isdir(species_path):
            continue

        image_count = sum(
            1 for fname in os.listdir(species_path)
            if fname.lower().endswith(image_extensions)
        )
        label_count = sum(
            1 for fname in os.listdir(species_path)
            if fname.lower().endswith(text_extensions)
        )
        print(f"  🏷️ Species: {species} - {image_count} images, {label_count} labels")
        class_total_images += image_count
        class_total_labels += label_count
    print(f"  🔢 Total for {class_name}: {class_total_images} images, {class_total_labels} labels")
    overall_total_images += class_total_images
    overall_total_labels += class_total_labels
print(f"\n🌟 Overall total images: {overall_total_images}")
print(f"📝 Overall total labels: {overall_total_labels}")
