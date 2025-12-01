import os
import torch
from ultralytics import YOLO

DATA_YAML   = r"E:/Santosh_master_thesis/DATA_YOLO_11_root_structure_leaves/yolo11s_seg_leaves.yaml"
START_MODEL = "yolo11s-seg.pt"

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
assert torch.cuda.is_available(), "CUDA is not available!"

def main():
    model = YOLO(START_MODEL)

    model.train(
        task="segment",
        data=DATA_YAML,
        imgsz=1024,
        epochs=120,
        batch=0,
        device=0,
        workers=0,

        retina_masks=True,

        # Species-safe augmentation
        mosaic=1.0,
        copy_paste=0.0,  # avoid cross-species mixing
        mixup=0.0,
        hsv_h=0.015, hsv_s=0.40, hsv_v=0.30,
        degrees=5, translate=0.07, scale=0.30, shear=3,
        flipud=0.05, fliplr=0.30,

        cos_lr=True,
        patience=20,

        project="species_segmentation_leaves",
        name="yolo11_leaves_seg_final",
        exist_ok=True,
    )

    model.val(
        data=DATA_YAML,
        imgsz=1024,
        conf=0.12,
        iou=0.80,
        max_det=300,
        retina_masks=True,
        plots=True
    )

if __name__ == "__main__":
    main()
