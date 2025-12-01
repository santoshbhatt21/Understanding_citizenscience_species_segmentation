import os
from pathlib import Path
import torch
from ultralytics import YOLO

# ── EDIT these paths if needed ─────────────────────────────────────────────────
DATA_YAML   = r"E:/Santosh_master_thesis/DATA_YOLO11_cleaned_labels/yolo11s_seg_10classes.yaml"
START_MODEL = r"yolo11m-seg.pt"   # or a previous best.pt

PROJECT_DIR = Path("segmentation_project_two_stage_cleaned_labels")
RUN_NAME_1  = "yolov11s_seg_1024_retina_v2"
RUN_NAME_2  = RUN_NAME_1 + "_ft_nomosaic"
F1_CONF     = 0.12               # use your F1-peak threshold
# ───────────────────────────────────────────────────────────────────────────────

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
assert torch.cuda.is_available(), "CUDA is not available. Please check your environment."

def main():
    # ── Stage 1: main training ────────────────────────────────────────────────
    model = YOLO(START_MODEL)
    results = model.train(
        task="segment",
        data=DATA_YAML,
        imgsz=1024,
        epochs=120,
        batch=0,                  # auto-batch
        device=0,
        workers=8,                # set 0 on Windows if you hit dataloader issues
        retina_masks=True,
        mosaic=1.0,
        copy_paste=0.30,
        hsv_h=0.015, hsv_s=0.70, hsv_v=0.40,
        degrees=10, translate=0.10, scale=0.50, shear=5,
        flipud=0.10, fliplr=0.50,
        mixup=0.10,
        cos_lr=True,
        patience=10,
        project=str(PROJECT_DIR),
        name=RUN_NAME_1,
        exist_ok=True,
    )

    # Resolve path to best weights from stage 1
    run_dir_1 = Path(getattr(results, "save_dir", PROJECT_DIR / RUN_NAME_1))
    best_pt_1 = run_dir_1 / "weights" / "best.pt"
    assert best_pt_1.exists(), f"best.pt not found at {best_pt_1}"

    # ── Stage 2: short fine-tune with no mosaic/copy-paste/mixup ──────────────
    ft = YOLO(str(best_pt_1))
    results_ft = ft.train(
        task="segment",
        data=DATA_YAML,
        imgsz=1024,
        epochs=15,
        batch=0,
        device=0,
        workers=8,
        retina_masks=True,
        mosaic=0.0,
        copy_paste=0.0,
        mixup=0.0,
        # keep light color/geom augs if you like; they rarely hurt:
        hsv_h=0.015, hsv_s=0.70, hsv_v=0.40,
        degrees=10, translate=0.10, scale=0.50, shear=5,
        flipud=0.10, fliplr=0.50,
        cos_lr=True,
        patience=5,
        project=str(PROJECT_DIR),
        name=RUN_NAME_2,
        exist_ok=True,
    )

    # Path to final best weights
    run_dir_2 = Path(getattr(results_ft, "save_dir", PROJECT_DIR / RUN_NAME_2))
    best_pt_2 = run_dir_2 / "weights" / "best.pt"
    assert best_pt_2.exists(), f"best.pt not found at {best_pt_2}"

    # ── Final validation at your F1-peak threshold ────────────────────────────
    ft.val(
        data=DATA_YAML,
        imgsz=1024,
        conf=F1_CONF,
        iou=0.80,
        max_det=300,
        retina_masks=True,
        plots=True,
        project=str(PROJECT_DIR),
        name=RUN_NAME_2 + "_val",
        exist_ok=True,
    )

if __name__ == "__main__":
    main()
