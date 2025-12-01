from ultralytics import YOLO
import cv2
import torch

# ---- Load your model ----
model = YOLO("E:/Santosh_master_thesis/species_segmentation/yolo11_10species_seg_final/weights/best.pt")  # path to your trained model

# ---- Path to new test image (outside training/val set) ----
img_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/TLS_IMAGES/250411_143717_06_01.jpg"  # Replace with your TLS image path

# ---- Run inference ----
results = model(img_path, save=True, conf=0.12, iou=0.5, show=True)

# ---- Access segmentation results ----
for result in results:
    print("Classes:", result.names)
    print("Boxes:", result.boxes.xyxy)
    print("Classes:", result.boxes.cls)
    print("Masks shape:", result.masks.data.shape if result.masks is not None else "No masks")
conf = results[0].boxes.conf[0].item()
print(f"Confidence: {conf:.2f}")
