log_path = "E:/Santosh_master_thesis/skipped_images.log"

with open(log_path, "r") as f:
    lines = f.readlines()

print(f"Total skipped images: {len(lines)}")