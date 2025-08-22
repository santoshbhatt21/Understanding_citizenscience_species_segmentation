import csv
from collections import Counter

# Path to predictions CSV
csv_path = "E:/Santosh_master_thesis/prediction_metadata_LOT.csv"

# Count images per predicted class only
class_counts = Counter()

with open(csv_path, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    if 'predicted_class' not in reader.fieldnames:
        raise KeyError("CSV must contain a 'predicted_class' column")
    for row in reader:
        cls = row['predicted_class']
        class_counts[cls] += 1

# Print per-class counts and total
total = 0
for cls in sorted(class_counts.keys()):
    n = class_counts[cls]
    print(f"{cls}: {n} images")
    total += n
print(f"\nTotal images: {total}")
