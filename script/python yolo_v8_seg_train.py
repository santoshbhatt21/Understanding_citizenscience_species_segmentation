import torch
from ultralytics import YOLO
import os
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
assert torch.cuda.is_available(), "CUDA is not available. Please check your environment."

# Path to your dataset YAML file
DATA_YAML = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/DATA_YOLOv8/data.yaml"
MODEL = "yolov8l-seg.pt"  # Consider using yolov8s or yolov8m for better performance

def main():
    model = YOLO(MODEL)
    model.train(
        data="E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/DATA_YOLOv8/data.yaml",
        epochs=30,
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
        project="segmentation_project_2",
        name="yolov8s_seg_balanced_512",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
