import os
import math
import random
from pathlib import Path
from glob import glob
import yaml
from collections import Counter

import torch
from ultralytics import YOLO

# ---------------- USER SETTINGS ----------------
# Use a 3-class YAML: names {0: Leaves, 1: Others, 2: Trunks}
DATA_YAML = r"E:/Santosh_master_thesis/DATA_YOLO11_LOT_all_three_classes_class_balance/data_cleaned.yaml"
# Train all three classes
CLASSES = [0, 1, 2]

PROJECT = "segmentation_project_01"
RUN_NAME = "yolo11s_seg_1024_L_O_T_balanced"

PRETRAINED = "yolo11s-seg.pt"
EPOCHS = 60                       # give the minority more passes
IMGSZ = 1024
BATCH = 12
WORKERS = 6

# Balancing configuration
# Equalize per-class image counts to the largest class (recommended)
BALANCE_ALL_CLASSES = True
# target per-class images ~= max_class_images * this ratio (used when PER_CLASS_CAP <= 0)
TARGET_RATIO_PER_CLASS = 1.0
# Hard per-class cap (exact target) for train list size per class
# Leaves, Others, Trunks will each be capped/upsampled to this many entries
PER_CLASS_CAP = 12000

# Or use single-minority upsampling (legacy)
MINORITY_CLASS = 2                # used only when BALANCE_ALL_CLASSES is False
# (minority : non-minority) desired in single-minority mode
TARGET_RATIO = 1.0
MAX_REPEAT = 8                    # safety cap for repeats of minority images
VAL_CONF = 0.001                  # keep very low for recall during val
PRED_CONF = 0.25                  # used only for the saved preview batch

# ------------------------------------------------

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
assert torch.cuda.is_available(), "CUDA not available"


def load_data_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    base = d.get("path", "")

    def _abs_path(x):
        if isinstance(x, str):
            return x if os.path.isabs(x) else os.path.join(base, x)
        return x

    # normalize split image dirs
    d["train"] = _abs_path(d["train"])  # images/train
    d["val"] = _abs_path(d["val"])      # images/val

    # images root (optional)
    d["images"] = _abs_path(d.get("images", os.path.join(base, "images")))

    # labels can be a root or a dict with train/val
    labels = d.get("labels")
    if isinstance(labels, dict):
        labels_train = _abs_path(labels.get(
            "train", os.path.join(base, "labels", "train")))
        labels_val = _abs_path(labels.get(
            "val", os.path.join(base, "labels", "val")))
    else:
        root = _abs_path(labels or os.path.join(base, "labels"))
        labels_train = os.path.join(root, "train")
        labels_val = os.path.join(root, "val")
    d["labels_train"], d["labels_val"] = labels_train, labels_val
    d["labels"] = {"train": labels_train, "val": labels_val}

    # preserve base path
    d["path"] = base
    return d


def list_images(root):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob(str(Path(root) / "**" / e), recursive=True))
    return files


def label_from_image(img_path, split_images_root, labels_root):
    """Map split images root/.../foo.jpg -> labels_root/.../foo.txt (mirrors tree)."""
    ip = Path(img_path)
    try:
        rel = ip.relative_to(split_images_root)
    except ValueError:
        # fall back: search by stem
        hit = list(Path(labels_root).rglob(ip.stem + ".txt"))
        return hit[0] if hit else None
    return Path(labels_root) / rel.with_suffix(".txt")


def has_class(lbl_path, wanted_cls):
    """Return True if txt file has at least one object of wanted_cls."""
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                cid = int(float(s.split()[0]))
                if cid == wanted_cls:
                    return True
    except Exception:
        pass
    return False


def count_objects(lbl_path):
    c = Counter()
    try:
        with open(lbl_path, "r", encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s:
                    continue
                cid = int(float(s.split()[0]))
                c[cid] += 1
    except Exception:
        pass
    return c


def build_balanced_train_file(data):
    """Create a train.txt with oversampling.
    - If BALANCE_ALL_CLASSES: equalize per-class image counts to the largest class.
    - Else: upsample a single minority class vs. the rest.
    """
    train_root = Path(data["train"])               # images/train folder
    labels_train_root = Path(data["labels_train"])  # labels/train folder

    # gather all train images
    train_imgs = list_images(train_root)

    obj_count = Counter()

    if BALANCE_ALL_CLASSES:
        per_class_imgs = {}
        image_classes = {}

        for img in train_imgs:
            lbl = label_from_image(img, train_root, labels_train_root)
            if not lbl or not lbl.exists():
                continue
            oc = count_objects(lbl)
            if oc:
                obj_count.update(oc)
            present = set(oc.keys())
            if not present:
                continue
            image_classes[img] = present
            for cid in present:
                per_class_imgs.setdefault(cid, []).append(img)

        if not per_class_imgs:
            print("WARNING: No labels found under train; using original train list.")
            balanced = train_imgs
            class_sizes = {}
            class_repeat = {}
        else:
            class_sizes = {int(c): int(len(imgs))
                           for c, imgs in per_class_imgs.items()}
            balanced = []
            final_sizes = {}
            if PER_CLASS_CAP and PER_CLASS_CAP > 0:
                # Exact per-class cap: downsample or oversample to PER_CLASS_CAP per class
                for c in [0, 1, 2]:
                    imgs = per_class_imgs.get(c, [])
                    n = len(imgs)
                    cap = PER_CLASS_CAP
                    if n == 0:
                        print(
                            f"WARNING: No images for class {c} in train; cannot reach cap {cap}.")
                        final_sizes[c] = 0
                        continue
                    if n >= cap:
                        chosen = random.sample(imgs, cap)
                        per_list = chosen
                    else:
                        q, r = divmod(cap, n)
                        per_list = imgs * q + \
                            (random.sample(imgs, r) if r > 0 else [])
                    final_sizes[c] = len(per_list)
                    balanced.extend(per_list)
            else:
                # Ratio-based equalization to max class size
                max_size = max(class_sizes.values()) if class_sizes else 0
                target = math.ceil(
                    max_size * TARGET_RATIO_PER_CLASS) if max_size else 0
                class_repeat = {}
                for c, size in class_sizes.items():
                    if size == 0:
                        class_repeat[c] = 0
                    elif target == 0:
                        class_repeat[c] = 1
                    else:
                        class_repeat[c] = min(MAX_REPEAT, max(
                            1, math.ceil(target / size)))
                # Per-image repeat = max repeat among its present classes
                for img, present in image_classes.items():
                    rep = max((class_repeat.get(c, 1)
                              for c in present), default=1)
                    balanced.extend([img] * rep)

        # Shuffle for better mixing
        random.shuffle(balanced)

        out_txt = (Path.cwd() / PROJECT / RUN_NAME /
                   "train_balanced.txt").resolve()
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            for p in balanced:
                f.write(str(p) + "\n")

        print("\n--- BALANCING SUMMARY (per-class) ---")
        print("Per-class original counts:", {int(k): int(v)
              for k, v in class_sizes.items()} if 'class_sizes' in locals() else {})
        if 'final_sizes' in locals() and final_sizes:
            print("Per-class final counts (capped/oversampled)",
                  {int(k): int(v) for k, v in final_sizes.items()})
        elif 'class_repeat' in locals():
            print("Per-class repeats:", {int(k): int(v)
                  for k, v in class_repeat.items()})
        print(f"Objects seen in labels  : {dict(obj_count)}")
        print(f"Balanced train list     : {len(balanced)} images")
        print(f"Wrote                   : {out_txt}\n")
        return str(out_txt)
    else:
        # Single-minority upsampling vs all others
        minority_imgs, non_minority_imgs = [], []
        for img in train_imgs:
            lbl = label_from_image(img, train_root, labels_train_root)
            if not lbl or not lbl.exists():
                continue
            oc = count_objects(lbl)
            obj_count.update(oc)
            if has_class(lbl, MINORITY_CLASS):
                minority_imgs.append(img)
            else:
                non_minority_imgs.append(img)

        n_non_minority = len(non_minority_imgs)
        n_minority = len(minority_imgs)

        if n_minority == 0:
            print(
                f"WARNING: No training images contain class {MINORITY_CLASS}. Using unbalanced train set.")
            repeat = 1
            balanced = non_minority_imgs
        else:
            repeat = min(MAX_REPEAT, max(1, math.ceil(
                TARGET_RATIO * max(1, n_non_minority) / max(1, n_minority))))
            balanced = non_minority_imgs + minority_imgs * repeat

        out_txt = (Path.cwd() / PROJECT / RUN_NAME /
                   "train_balanced.txt").resolve()
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(out_txt, "w", encoding="utf-8") as f:
            for p in balanced:
                f.write(str(p) + "\n")

        print("\n--- BALANCING SUMMARY (single-minority) ---")
        print(f"Non-minority images     : {n_non_minority}")
        print(f"Minority images (class {MINORITY_CLASS}): {n_minority}")
        print(f"Repeat factor for minority: {repeat} (cap {MAX_REPEAT})")
        print(f"Objects seen in labels  : {dict(obj_count)}")
        print(f"Balanced train list     : {len(balanced)} images")
        print(f"Wrote                   : {out_txt}\n")
        return str(out_txt)


def main():
    data = load_data_yaml(DATA_YAML)
    # build balanced train file (oversample Trunks)
    train_txt = build_balanced_train_file(data)

    # Prepare names/nc from YAML; ensure 3-class setup
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    names = raw_cfg.get("names")
    if isinstance(names, dict):
        keys = [int(k) for k in names.keys()]
        max_id = max(keys) if keys else -1
        names_list = [names.get(i, names.get(
            str(i), f"cls{i}")) for i in range(max_id + 1)]
    else:
        names_list = list(names) if names else []
    if len(names_list) < 3:
        raise RuntimeError(
            "DATA_YAML must define 3 classes (Leaves, Others, Trunks). Update DATA_YAML and retry.")

    # Write a derived YAML that keeps labels mapping but swaps train to our list
    out_yaml = Path(PROJECT) / RUN_NAME / "data_balanced.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_cfg = dict(raw_cfg)
    out_cfg["train"] = train_txt
    # ensure nc/names correct
    out_cfg["names"] = names
    out_cfg["nc"] = len(names_list)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_cfg, f, sort_keys=False)

    model = YOLO(PRETRAINED)

    results = model.train(
        data=str(out_yaml),
        classes=CLASSES,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=0,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        # === imbalance-friendly knobs ===
        # augmentations
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
        degrees=10, translate=0.08, scale=0.4, shear=5,
        flipud=0.0, fliplr=0.0,
        mosaic=0.2,                 # a bit more to expose trunks in context
        copy_paste=0.3,             # helps minority by pasting objects
        mixup=0.05,
        retina_masks=True,
        mask_ratio=2,
        optimizer="AdamW",
        lr0=0.0006, lrf=0.12, cos_lr=True, warmup_epochs=3,
        patience=20,
        cache="disk",
        plots=False
    )

    # Validate with a low conf to expose recall
    best = Path(results.save_dir) / "weights" / "best.pt"
    assert best.exists(), f"best.pt not found at {best}"
    model = YOLO(str(best))
    model.val(
        data=str(out_yaml),
        classes=CLASSES,
        imgsz=IMGSZ,
        device=0,
        workers=WORKERS,
        split="val",
        conf=VAL_CONF,   # <-- low conf to measure recall fairly
        iou=0.6,
        max_det=3000,
        rect=True,
        half=True,
        plots=True,
    )

    # (Optional) Save a quick preview grid at a higher conf
    model.predict(
        source=data["val"],
        imgsz=IMGSZ,
        conf=PRED_CONF,
        iou=0.6,
        device=0,
        save=True,
        project=str(results.save_dir),
        name="pred_samples",
        exist_ok=True,
        classes=CLASSES,
        max_det=3000,
        verbose=False,
    )


if __name__ == "__main__":
    main()
