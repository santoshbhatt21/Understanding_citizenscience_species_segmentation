import os
import shutil
from pathlib import Path

def flatten_folder(input_dir, output_dir, exts=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if exts is not None:
        exts = [e.lower() for e in exts]

    for subdir, _, files in os.walk(input_dir):
        for file in files:
            src_file = Path(subdir) / file
            if exts is None or src_file.suffix.lower() in exts:
                relative = src_file.relative_to(input_dir)
                new_name = "_".join(relative.parts)
                dst_file = output_dir / new_name
                shutil.copy2(src_file, dst_file)

# Flatten images
flatten_folder("YOLO/images/train", "YOLO/flat/images/train", exts=[".jpg", ".jpeg", ".png"])
flatten_folder("YOLO/images/val", "YOLO/flat/images/val", exts=[".jpg", ".jpeg", ".png"])

# Flatten labels (.txt YOLO polygons)
flatten_folder("YOLO/labels/train", "YOLO/flat/labels/train", exts=[".txt"])
flatten_folder("YOLO/labels/val", "YOLO/flat/labels/val", exts=[".txt"])