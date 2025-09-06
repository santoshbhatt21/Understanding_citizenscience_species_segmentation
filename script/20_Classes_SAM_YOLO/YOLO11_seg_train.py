import torch
import os
import argparse
from pathlib import Path
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
assert torch.cuda.is_available(), "CUDA is not available."

# Defaults
DEFAULT_MODEL = "yolo11m-seg.pt"  # stronger backbone for small objects
DEFAULT_DATA1 = r"E:/Santosh_master_thesis/DATA_YOLO11_strict_CAM_setting/data_20_classes_crops.yaml"
DEFAULT_DATA2 = r"E:/Santosh_master_thesis/DATA_YOLO11_strict_CAM_setting/data_20_classes_mixed.yaml"


def add_common_hyp(kwargs: dict):
    # Foreground-friendly loader & a balanced augmentation set
    kwargs.update(
        dict(
            imgsz=960,
            batch=0,
            device=0,
            seed=42,
            cache=False,  # avoid writing/reading .cache while dataset stabilizes
            rect=True,
            multi_scale=True,
            close_mosaic=10,
            mosaic=0.5,
            mixup=0.0,
            copy_paste=0.3,
            perspective=0.0,
            scale=0.4,
            degrees=10,
            shear=5,
            translate=0.05,
            fliplr=0.5,
            flipud=0.2,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            cls=1.5,
            box=7.5,
            dfl=1.0,
            optimizer="AdamW",
            lr0=0.0008,
            lrf=0.1,
            cos_lr=True,
            warmup_epochs=5,
            patience=20,
            workers=0,    # Windows: avoid multiprocessing pickling issues
            project="segmentation_project_20_classes_drop_empty_and_crop",
            exist_ok=True,
        )
    )
    return kwargs


def parse_args():
    ap = argparse.ArgumentParser(
        description="Two-stage YOLO11 seg training: crops -> mixed")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="Initial model weights (e.g., yolo11m-seg.pt)")
    ap.add_argument("--data1", default=DEFAULT_DATA1,
                    help="Phase-1 data YAML (crops)")
    ap.add_argument("--epochs1", type=int, default=0,
                    help="Epochs for phase 1")
    ap.add_argument("--data2", default=DEFAULT_DATA2,
                    help="Phase-2 data YAML (mixed)")
    ap.add_argument("--epochs2", type=int, default=80,
                    help="Epochs for phase 2")
    ap.add_argument(
        "--name1", default="yolo11m_seg_fg_biased_960_stage1_crops", help="Run name for phase 1")
    ap.add_argument(
        "--name2", default="yolo11m_seg_fg_biased_960_stage2_mixed", help="Run name for phase 2")
    return ap.parse_args()


def main():
    args = parse_args()

    best_weights = None
    if args.epochs1 > 0:
        # Phase 1: train on crops
        model = YOLO(args.model)
        train_kwargs = add_common_hyp(
            dict(data=args.data1, epochs=args.epochs1, name=args.name1))
        res1 = model.train(**train_kwargs)

        # Locate best weights from phase 1
        save_dir = Path(getattr(res1, "save_dir", Path(
            "runs/segment/train") / args.name1))
        best = save_dir / "weights" / "best.pt"
        if best.exists():
            best_weights = best
        else:
            last = save_dir / "weights" / "last.pt"
            best_weights = last if last.exists() else None
    else:
        print("[INFO] Skipping phase 1 (epochs1=0). Starting phase 2 from base model.")

    # Phase 2: continue on mixed
    start_weights = str(best_weights) if isinstance(
        best_weights, Path) and best_weights.exists() else args.model
    model2 = YOLO(start_weights)
    train_kwargs2 = add_common_hyp(
        dict(data=args.data2, epochs=args.epochs2, name=args.name2))
    model2.train(**train_kwargs2)


if __name__ == "__main__":
    main()
