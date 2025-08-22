import os
from collections import Counter

# Quick check: show class mapping, raw counts, and target train/val counts (80/20 of 3000)
root = r"E:/Santosh_master_thesis/LOT_all_images_labeled"
if not os.path.isdir(root):
    print(f"Dataset folder not found: {root}")
    raise SystemExit(1)

classes = sorted([d for d in os.listdir(
    root) if os.path.isdir(os.path.join(root, d))])
class_to_idx = {c: i for i, c in enumerate(classes)}
counts = Counter()
for i, c in enumerate(classes):
    cpath = os.path.join(root, c)
    for dirpath, dirnames, filenames in os.walk(cpath):
        dirnames.sort()
        filenames = [f for f in filenames if f.lower().endswith(
            (".jpg", ".jpeg", ".png"))]
        counts[i] += len(filenames)

print("Class to idx mapping:", class_to_idx)
print("Raw images per class:", counts)

TARGET = 3000
train_target = int(0.8 * TARGET)
val_target = TARGET - train_target
train_counts = Counter(
    {i: (train_target if counts[i] > 0 else 0) for i in class_to_idx.values()})
val_counts = Counter(
    {i: (val_target if counts[i] > 0 else 0) for i in class_to_idx.values()})
print(f"Train class counts (target ~80% of {TARGET} ): {train_counts}")
print(f"Val class counts (target ~20% of {TARGET} ): {val_counts}")
