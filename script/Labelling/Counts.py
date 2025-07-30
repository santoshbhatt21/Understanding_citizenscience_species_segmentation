import os

# === Root path to your sorted dataset ===
root_dir = "E:/Santosh_master_thesis/Species_folder_sorted_images"

# === Define image extensions to count ===
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
overall_total = 0

# === Traverse classes and species subfolders ===
for class_name in sorted(os.listdir(root_dir)):
    class_path = os.path.join(root_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"\n📂 Class: {class_name}")
    class_total = 0
    for species in sorted(os.listdir(class_path)):
        species_path = os.path.join(class_path, species)
        if not os.path.isdir(species_path):
            continue

        image_count = sum(
            1 for fname in os.listdir(species_path)
            if fname.lower().endswith(image_extensions)
        )
        print(f"  🏷️ Species: {species} - {image_count} images")
        class_total += image_count
    print(f"  🔢 Total for {class_name}: {class_total} images")
    overall_total += class_total
    print(f"\n🌟 Overall total images: {overall_total}") 