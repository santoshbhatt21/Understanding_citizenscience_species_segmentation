import glob, os

images_root = r"E:\Santosh_master_thesis\DATA_YOLO11_strict_CAM_setting\images"
labels_root = r"E:\Santosh_master_thesis\DATA_YOLO11_strict_CAM_setting\labels"

missing = []
for img in glob.glob(os.path.join(images_root, "**", "*.jpg"), recursive=True):
    base = os.path.splitext(os.path.basename(img))[0]
    label = glob.glob(os.path.join(labels_root, "**", base + ".txt"), recursive=True)
    if not label:
        missing.append(img)

print("Images missing labels:", len(missing))
print("Examples:", missing[:10])
