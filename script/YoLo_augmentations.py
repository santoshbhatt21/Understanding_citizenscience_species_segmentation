import torch
from ultralytics import YOLO
import os
from collections import Counter

def get_class_counts(label_dir):
    counts = Counter()
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), "r") as f:
                for line in f:
                    cls = int(line.strip().split()[0])
                    counts[cls] += 1
    return counts

def main():
    assert torch.cuda.is_available(), "CUDA is not available. Please check your environment."

    # Example manual class weights (not used by YOLOv8)
    class_weights = [0.2, 1.0, 1.0, 1.0, 1.0, 1]

    # Train the model
    model = YOLO("yolov8s-seg.pt")
    model.train(
        data="E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/YOLO/flat/data.yaml",
        epochs=50,
        imgsz=512,
        device=0,
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=20,
        translate=0.1,
        scale=0.5,
        shear=10,
        flipud=0.3,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.2,
        # Optional settings
        patience=10,
        workers=4,
        project="segmentation_project",
        name="yolov8s_seg_balanced_512",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()