import os
from pathlib import Path
from datetime import datetime
import random
import json
from glob import glob

import torch
from ultralytics import YOLO
import yaml

# --------- YOLOv11 segmentation training (supports all classes) ---------
# Uses your dataset YAML directly; optionally filter a subset via CLASSES.

# Point to your existing dataset YAML (keeps original images, cleaned labels)
DATA_YAML = r"E:/Santosh_master_thesis/DATA_YOLO11_strict_CAM_setting/data_20_classes.yaml"

# Train/eval only on these classes; set to None to use ALL classes in the YAML
CLASSES = None

# ---------- Imbalance handling (optional, no dataset changes) ----------
# If enabled, we will generate a balanced train file list by oversampling
# images that contain the minority class.
ENABLE_CLASS_BALANCING = False  # two-class-only logic; leave False for 20 classes
MINORITY_CLASS = 1  # legacy (used only if balancing enabled)
OVERSAMPLE_FACTOR = 4
INCLUDE_MAJORITY_ONCE = True  # keep other images once for variety
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")

# Weights and run settings
PRETRAINED = "yolo11s-seg.pt"
PROJECT = "segmentation_project_yolo11"
RUN_NAME = "yolo11s_seg_896_LT_20_classes"

EPOCHS = 40
IMGSZ_TRAIN = 1024
IMGSZ_VAL = 1024
WORKERS = 6
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


def _resolve_path(base: str, entry):
    # entry can be absolute path string, relative string, or a list of strings
    if isinstance(entry, str):
        return [entry if os.path.isabs(entry) else os.path.join(base, entry)]
    elif isinstance(entry, list):
        out = []
        for e in entry:
            out.append(e if os.path.isabs(e) else os.path.join(base, e))
        return out
    else:
        return []


def _labels_train_root(data_cfg: dict, base: str) -> str:
    labels_entry = data_cfg.get("labels")
    if isinstance(labels_entry, dict):
        v = labels_entry.get("train")
        if v:
            return v if os.path.isabs(v) else os.path.join(base, v)
    if isinstance(labels_entry, str):
        return labels_entry if os.path.isabs(labels_entry) else os.path.join(base, labels_entry)
    # default convention
    return os.path.join(base, "labels", "train")


def _map_image_to_label(img_path: str, images_roots: list, labels_root: str) -> str:
    p = Path(img_path)
    for ir in images_roots:
        try:
            rel = p.relative_to(ir)
            return str(Path(labels_root) / rel.with_suffix(".txt"))
        except Exception:
            continue
    # fallback: replace images->labels in path string
    s = str(p)
    s = s.replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")
    return str(Path(s).with_suffix(".txt"))


def _has_class(label_file: str, cls_id: int) -> bool:
    try:
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == str(cls_id):
                    return True
    except Exception:
        return False
    return False


def _build_balanced_train_list(data_yaml_path: str, save_dir: Path):
    # Read dataset config
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    base = data_cfg.get("path", "")
    images_train_roots = _resolve_path(base, data_cfg.get("train"))
    labels_train_root = _labels_train_root(data_cfg, base)

    # All train images
    all_train_images = []
    for ir in images_train_roots:
        for ext in IMG_EXTS:
            all_train_images.extend(
                glob(str(Path(ir) / "**" / ext), recursive=True))

    # Split into minority-containing and others
    minority_imgs = []
    others = []
    for img in all_train_images:
        lbl = _map_image_to_label(img, images_train_roots, labels_train_root)
        if _has_class(lbl, MINORITY_CLASS):
            minority_imgs.append(img)
        else:
            if INCLUDE_MAJORITY_ONCE:
                others.append(img)

    # Oversample
    balanced = []
    for _ in range(max(1, OVERSAMPLE_FACTOR)):
        balanced.extend(minority_imgs)
    balanced.extend(others)

    random.shuffle(balanced)

    # Write list and derived YAML
    list_path = save_dir / "balanced_train.txt"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(balanced) + "\n", encoding="utf-8")

    balanced_yaml = save_dir / "data_balanced.yaml"
    out_cfg = dict(data_cfg)
    out_cfg["train"] = str(list_path)
    with open(balanced_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    return str(balanced_yaml), {
        "total_images": len(all_train_images),
        "minority_images": len(minority_imgs),
        "others_images": len(others),
        "oversample_factor": OVERSAMPLE_FACTOR,
        "final_train_count": len(balanced),
    }


def main():
    model = YOLO(PRETRAINED)

    effective_run_name = _choose_effective_run_name(RUN_NAME, PROJECT)
    save_dir = Path(PROJECT) / effective_run_name
    best = save_dir / "weights" / "best.pt"

    # augmentations / schedule consistent with your previous runs
    hsv_h, hsv_s, hsv_v = (0.015, 0.5, 0.3)
    degrees, translate, scale, shear = (10, 0.08, 0.4, 5)
    flipud, fliplr = (0.0, 0.0)
    mosaic = 0.1
    mixup = 0.0
    copy_paste = 0.1
    close_mosaic = 0
    freeze = 0

    # Optionally build a balanced train filelist (no dataset changes)
    data_for_train = DATA_YAML
    if ENABLE_CLASS_BALANCING:
        try:
            data_for_train, balance_stats = _build_balanced_train_list(
                DATA_YAML, save_dir)
            print("Class balancing enabled:", balance_stats)
            if balance_stats.get("minority_images", 0) == 0:
                print(
                    "No minority-class images found in train. Falling back to original train set.")
                data_for_train = DATA_YAML
        except Exception as e:
            print("Balancing skipped due to error:", e)
            data_for_train = DATA_YAML

    # ---------------- train ----------------
    train_kwargs = dict(
        data=data_for_train,
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
        lr0=0.0002,
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
        patience=15,
        freeze=freeze,
        fraction=1.0,
    )
    if CLASSES is not None:
        train_kwargs["classes"] = CLASSES
    model.train(**train_kwargs)

    assert best.exists(), f"best.pt not found at {best}"

    # ---------------- validate (plots ON) ----------------
    model = YOLO(str(best))
    val_kwargs = dict(
        data=DATA_YAML,
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
    if CLASSES is not None:
        val_kwargs["classes"] = CLASSES
    metrics = model.val(**val_kwargs)
    results = metrics.results_dict
    print("Final metrics:", results)

    # Save quick macro F1 from boxes
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
    (save_dir / "metrics_extra.json").write_text(json.dumps(
        {"F1_boxes": f1_score, "classes": (CLASSES if CLASSES is not None else "all")}, indent=2))

    # Save a few preds from val for inspection
    try:
        with open(DATA_YAML, "r") as f:
            data_cfg = yaml.safe_load(f)
        base = data_cfg.get("path", "")
        val_entry = data_cfg.get("val")
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
            pred_kwargs = dict(
                source=sample_imgs,
                imgsz=IMGSZ_VAL,
                conf=0.25,
                iou=0.6,
                device=0,
                save=True,
                project=str(save_dir),
                name="pred_samples",
                exist_ok=True,
                max_det=3000,
                verbose=False,
            )
            if CLASSES is not None:
                pred_kwargs["classes"] = CLASSES
            _ = model.predict(**pred_kwargs)
    except Exception as e:
        print("Prediction export skipped:", e)


if __name__ == "__main__":
    main()
