import os

root_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labelling_Manual"
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
label_extensions = ('.txt',)

overall_total = 0
counted_type = {"images": 0, "labels": 0}

for species in sorted(os.listdir(root_dir)):
    species_path = os.path.join(root_dir, species)
    if not os.path.isdir(species_path):
        continue

    files = os.listdir(species_path)
    image_count = sum(1 for fname in files if fname.lower().endswith(image_extensions))

    if image_count > 0:
        print(f"{species}: {image_count} images")
        overall_total += image_count
        counted_type["images"] += image_count
    else:
        label_count = sum(1 for fname in files if fname.lower().endswith(label_extensions))
        print(f"{species}: {label_count} labels")
        overall_total += label_count
        counted_type["labels"] += label_count

print("\nSummary:")
print(f"  Counted images: {counted_type['images']}")
print(f"  Counted labels: {counted_type['labels']}")
print(f"🌟 Total counted items across species folders: {overall_total}")