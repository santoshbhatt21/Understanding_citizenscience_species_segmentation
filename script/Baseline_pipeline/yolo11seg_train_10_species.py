import os
import torch
from ultralytics import YOLO

# EDIT these paths if needed
DATA_YAML   = r"E:/Santosh_master_thesis/DATA_YOLO_11_root_structure/yolo11s_seg_10classes.yaml"
START_MODEL = r"yolo11m-seg.pt"    # or your previous best.pt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
assert torch.cuda.is_available(), "CUDA is not available. Please check your environment."

def main():
    model = YOLO(START_MODEL)
    model.train(
        task="segment",
        data=DATA_YAML,
        imgsz=1024,
        epochs=120,
        batch=0,                 # <-- auto-batch; was "auto"
        device=0,
        workers=8,               # if Windows gives issues, set to 0
        retina_masks=True,
        mosaic=1.0,
        copy_paste=0.30,
        hsv_h=0.015, hsv_s=0.70, hsv_v=0.40,
        degrees=10, translate=0.10, scale=0.50, shear=5,
        flipud=0.10, fliplr=0.50,
        mixup=0.10,
        cos_lr=True,
        patience=20,
        project="segmentation_project_10_species_cleaned_labels",
        name="yolov11s_seg_1024_retina_v2",
        exist_ok=True,
    )

    # Optional: validate right after training at the F1-peak threshold you found
    model.val(data=DATA_YAML, imgsz=1024, conf=0.12, iou=0.80, max_det=300, retina_masks=True, plots=True)

if __name__ == "__main__":
    main()
