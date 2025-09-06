import os
import json
import random
from glob import glob
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO
import yaml

# Train YOLOv11 segmentation on a subset of classes: Leaves (0) and Trunks (2)
# Dataset stays the same; we filter classes during training/eval.

# uses same images, cleaned labels
DATA_YAML = r"E:/Santosh_master_thesis/DATA_YOLOv8/data_cleaned.yaml"
PRETRAINED = "yolo11s-seg.pt"
PROJECT = "segmentation_project_0"
RUN_NAME = "yolo11s_seg_896_leaves_trunks"

# Only these classes from the dataset names: 0=Leaves, 2=Trunks (ignore class 1=Others)
CLASSES = [0, 2]

EPOCHS = 40
IMGSZ_TRAIN = 896
IMGSZ_VAL = 896
WORKERS = 2
BATCH = 12

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
assert torch.cuda.is_available(), "CUDA not available"


def _can_open_for_append(p: Path) -> bool:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8"):
            pass
        return True
    except PermissionError:
        return False
    except Exception:
        return True


def _choose_effective_run_name(base_name: str, project: str) -> str:
    save_dir = Path(project) / base_name
    results_csv = save_dir / "results.csv"
    if results_csv.exists() and not _can_open_for_append(results_csv):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{base_name}_{ts}"
    if not _can_open_for_append(results_csv):
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{base_name}_{ts}"
    return base_name


def main():
    model = YOLO(PRETRAINED)

    effective_run_name = _choose_effective_run_name(RUN_NAME, PROJECT)
    save_dir = Path(PROJECT) / effective_run_name
    best = save_dir / "weights" / "best.pt"

    # Augs/LR same as your baseline
    hsv_h, hsv_s, hsv_v = (0.015, 0.5, 0.3)
    degrees, translate, scale, shear = (10, 0.08, 0.4, 5)
    flipud, fliplr = (0.0, 0.0)
    mosaic = 0.2
    mixup = 0.05
    copy_paste = 0.2
    close_mosaic = 0
    freeze = 0

    # Train only on the selected classes
    model.train(
        data=DATA_YAML,
        classes=CLASSES,
        epochs=EPOCHS,
        imgsz=IMGSZ_TRAIN,
        batch=BATCH,
        device=0,
        workers=WORKERS,
        project=PROJECT,
        name=effective_run_name,
        exist_ok=True,
        plots=False,
        optimizer="AdamW",
        lr0=0.0015,
        lrf=0.12,
        cos_lr=True,
        warmup_epochs=3,
        hsv_h=hsv_h, hsv_s=hsv_s, hsv_v=hsv_v,
        degrees=degrees, translate=translate, scale=scale, shear=shear,
        flipud=flipud, fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        copy_paste=copy_paste,
        close_mosaic=close_mosaic,
        multi_scale=False,
        cache='disk',
        retina_masks=True,
        mask_ratio=2,
        patience=8,
        freeze=freeze,
        fraction=1.0,
    )

    assert best.exists(), f"best.pt not found at {best}"

    # Final validation (on the same two classes)
    model = YOLO(str(best))
    metrics = model.val(
        data=DATA_YAML,
        classes=CLASSES,
        imgsz=IMGSZ_VAL,
        device=0,
        workers=WORKERS,
        split="val",
        conf=0.001,
        iou=0.6,
        max_det=3000,
        plots=True,
        save_json=False,
        rect=True,
        half=True,
    )
    results = metrics.results_dict
    print("Final metrics (Leaves, Trunks only):", results)

    # Compute a macro F1 from the reported aggregate P/R (boxes)
    def find_key(d, contains: str):
        for k in d.keys():
            if contains in k:
                return k
        return None

    p_key = find_key(results, "precision(B)") or find_key(results, "precision")
    r_key = find_key(results, "recall(B)") or find_key(results, "recall")
    f1_score = None
    if p_key and r_key:
        P, R = float(results[p_key]), float(results[r_key])
        denom = (P + R) if (P + R) > 0 else 1e-12
        f1_score = 2 * P * R / denom
        print(f"Macro F1 (boxes): {f1_score:.4f}")

    extra = {"F1_boxes": f1_score, "precision_key": p_key,
             "recall_key": r_key, "classes": CLASSES}
    (save_dir / "metrics_extra.json").write_text(json.dumps(extra, indent=2))

    # Save a small batch of predictions from val
    try:
        with open(DATA_YAML, "r") as f:
            data_cfg = yaml.safe_load(f)
        val_entry = data_cfg.get("val")
        base = data_cfg.get("path", "")
        val_paths = []
        if isinstance(val_entry, str):
            val_paths = [os.path.join(base, val_entry) if not os.path.isabs(
                val_entry) else val_entry]
        elif isinstance(val_entry, list):
            for ve in val_entry:
                val_paths.append(os.path.join(base, ve)
                                 if not os.path.isabs(ve) else ve)

        img_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        all_val_images = []
        for vp in val_paths:
            for ext in img_exts:
                all_val_images.extend(
                    glob(str(Path(vp) / "**" / ext), recursive=True))

        random.shuffle(all_val_images)
        sample_imgs = all_val_images[:24] if len(
            all_val_images) > 24 else all_val_images
        if sample_imgs:
            pred_out = str(save_dir / "pred_samples")
            _ = model.predict(
                source=sample_imgs,
                imgsz=IMGSZ_VAL,
                conf=0.25,
                iou=0.6,
                device=0,
                save=True,
                project=str(save_dir),
                name="pred_samples",
                exist_ok=True,
                stream=False,
                max_det=3000,
                verbose=False,
                classes=CLASSES,
            )
            print(f"Saved batch predictions to: {pred_out}")
        else:
            print("No validation images found for batch predictions.")
    except Exception as e:
        print(f"Batch prediction step skipped due to error: {e}")


if __name__ == "__main__":
    main()
