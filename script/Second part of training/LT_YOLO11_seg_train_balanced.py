#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LT_YOLO11_seg_train_balanced.py
--------------------------------
YOLO11 segmentation training with **multi-class balancing**, **focal loss**, and
**per-class segmentation AP export**.

New in this version
- You can train by passing either a ready **data.yaml** OR just a **--root** folder.
- When using --root, the script auto-builds a minimal data.yaml (with train/val subdirs),
  and will **infer class count** from labels if you don't provide --names.

Dataset expectations (when using --root)
- <root>/images/train/**.jpg|png
- <root>/labels/train/**.txt    (YOLO polygon format)
- <root>/images/val/**.jpg|png
- <root>/labels/val/**.txt
- Optional: <root>/names.txt (one class per line) if you don't want auto-inferred names.

Usage examples (PowerShell)
---------------------------
# A) Use an existing data.yaml (classic mode)
python LT_YOLO11_seg_train_balanced.py --data "E:/dataset/data.yaml" --project "E:/runs" --name "exp1"

# B) Use only a dataset root (no YAML needed)
python LT_YOLO11_seg_train_balanced.py --root "E:/dataset" --project "E:/runs" --name "exp2"

# B.1) If your subfolders differ, set them:
python LT_YOLO11_seg_train_balanced.py --root "E:/dataset" --train-subdir "imgs/train" --val-subdir "imgs/val" --project "E:/runs"

# B.2) Provide class names explicitly (comma-separated) or from a file
python LT_YOLO11_seg_train_balanced.py --root "E:/dataset" --names "oak,beech,spruce"
python LT_YOLO11_seg_train_balanced.py --root "E:/dataset" --names-file "E:/dataset/names.txt"
"""

import os
import json
import math
import random
import argparse
from pathlib import Path
from glob import glob
from typing import List, Optional, Tuple, Dict
import yaml

# lazy import ultralytics so --help works without it installed


def _lazy_ultralytics():
    from ultralytics import YOLO
    return YOLO


# Mitigate Windows/OpenMP runtime conflicts (Intel libiomp5 duplication) and over-threading
# Must be set BEFORE heavy libs (torch/onnxruntime/opencv) are imported by ultralytics
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif",
            "*.tiff", "*.JPG", "*.PNG", "*.JPEG")

# ------------------------------
# USER CONFIG (used when no CLI args are provided)
# ------------------------------
USER_CONFIG = {
    # If you prefer not to pass CLI args, set these and run the script directly
    # True to auto-generate data.yaml from root; False to use explicit data.yaml
    "use_root": True,
    "root": r"E:/Santosh_master_thesis/DATA_YOLO11_classified_Leaves_Trunks",
    "data": None,  # set to a data.yaml path if use_root=False
    "project": r"E:/Santosh_master_thesis/Checkpoints_YOLO11_seg",
    "name": "yolo11seg_manual_run",
    "imgsz": 640,
    "epochs": 40,
    "batch": 32,
    "seed": 42,
    "workers": 6,
    "model": r"E:/Santosh_master_thesis/yolo11n-seg.pt",
    # balancing
    "no_balance": False,
    "balance_target": None,
    "balance_mult": 0.6,
    # data.yaml from root
    "train_subdir": "images/train",
    "val_subdir": "images/val",
    "names": None,  # comma-separated string or None
    "names_file": None,
    # train knobs (match CLI defaults)
    "lr0": 0.005,
    "lrf": 0.05,
    "retina_masks": True,
    "max_det": 300,
    "fraction": 1.0,
    "freeze": None,
    "rect": True,
    "resume": False,
    "auto_augment": "randaugment",
    "erasing": 0.4,
    "iou": 0.7,
    "kobj": 1.0,
}

# ------------------------------
# YAML + path helpers
# ------------------------------


def _read_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def _resolve_to_list(base: str, node) -> List[str]:
    """
    Resolve a data.yaml 'train' or 'val' node into a list of absolute paths.
    node can be: str (dir, file, or txt list), or list[str].
    """
    paths = []
    parts = node if isinstance(node, list) else [node]
    for p in parts:
        if p is None:
            continue
        p = str(p)
        p_abs = os.path.join(base, p) if base and not os.path.isabs(p) else p
        paths.append(os.path.normpath(p_abs))
    return paths


def _expand_image_dirs(dirs: List[str]) -> List[str]:
    """Expand list of image directories into file paths (absolute)."""
    out = []
    for d in dirs:
        if os.path.isdir(d):
            for ext in IMG_EXTS:
                out.extend(glob(str(Path(d) / "**" / ext), recursive=True))
        elif os.path.isfile(d) and d.lower().endswith(".txt"):
            with open(d, "r", encoding="utf-8") as f:
                out.extend([line.strip() for line in f if line.strip()])
        else:
            if os.path.isfile(d):
                out.append(d)
    out = sorted(list({os.path.normpath(p) for p in out}))
    return out


def _derive_labels_roots(train_nodes, base) -> List[str]:
    """
    Try to derive candidate labels roots for mapping images->labels.
    For a path like ".../images/train", propose ".../labels/train".
    """
    labels_roots = set()
    for p in _resolve_to_list(base, train_nodes):
        path = Path(p)
        s = str(path).replace("\\", "/")
        if "/images/" in s:
            labels_roots.add(s.replace("/images/", "/labels/"))
        elif s.endswith("/images") or s.endswith("\\images"):
            labels_roots.add(s[:-7] + "labels")
        if path.name.lower() == "images":
            labels_roots.add(str(path.with_name("labels")))
    return [os.path.normpath(x) for x in labels_roots]


def _image_to_label(img_path: str, image_roots: List[str], labels_roots: List[str]) -> Path:
    """
    Heuristic mapping: <root>/images/train/aa/bb.jpg -> <root>/labels/train/aa/bb.txt
    """
    ip = Path(img_path)
    ip_norm = os.path.normpath(str(ip))
    for ir in image_roots:
        ir_norm = os.path.normpath(ir)
        if ip_norm.startswith(ir_norm):
            rel = os.path.relpath(ip_norm, ir_norm)
            for lr in labels_roots:
                lp = Path(lr) / Path(rel).with_suffix(".txt")
                return lp
    s = str(ip).replace("\\", "/")
    if "/images/" in s:
        return Path(s.replace("/images/", "/labels/")).with_suffix(".txt")
    return Path(ip).with_suffix(".txt")

# ------------------------------
# Labels sanity check
# ------------------------------


def check_label_sanity(data_yaml_path: str, sample: int = 300) -> Dict[str, int]:
    """
    Quick scan of label files to catch common issues:
      - non-numeric tokens
      - coords outside [0,1]
      - too-short polygons (< 4 points)
    Returns a dict of counters and prints a short summary.
    """
    cfg = _read_yaml(data_yaml_path)
    base = cfg.get("path", "")
    train_nodes = cfg.get("train")
    if train_nodes is None:
        raise ValueError("data.yaml missing 'train' key")
    image_roots = _resolve_to_list(base, train_nodes)
    labels_roots = _derive_labels_roots(train_nodes, base)
    images = _expand_image_dirs(image_roots)
    stats = dict(files_scanned=0, bad_tokens=0,
                 out_of_range=0, short_polys=0, empty=0)
    for img in images[:sample]:
        lbl_path = _image_to_label(img, image_roots, labels_roots)
        if not lbl_path.is_file():
            continue
        stats["files_scanned"] += 1
        try:
            with open(lbl_path, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
        except Exception:
            continue
        if not lines:
            stats["empty"] += 1
            continue
        for ln in lines:
            parts = ln.split()
            # need at least class + 4 coords (2 points), but seg usually has many more
            if len(parts) < 5:
                stats["short_polys"] += 1
                continue
            try:
                floats = [float(x) for x in parts]
            except Exception:
                stats["bad_tokens"] += 1
                continue
            # coords must be in [0,1]
            coords = floats[1:]
            if any((c < 0.0 or c > 1.0) for c in coords):
                stats["out_of_range"] += 1
    print("\n[label check] sample scan:")
    print(json.dumps(stats, indent=2))
    if stats["out_of_range"] > 0:
        print(
            "WARNING: Found coords outside [0,1]. Labels may be unnormalized.")
    if stats["bad_tokens"] > 0:
        print("WARNING: Found non-numeric tokens in labels.")
    if stats["short_polys"] > 0:
        print("WARNING: Found very short polygons. Confirm polygon format.")
    return stats

# ------------------------------
# Building data from --root
# ------------------------------


def _read_names_file(names_file: Path) -> List[str]:
    with open(names_file, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines


def _scan_max_class_id(labels_root: Path) -> int:
    """Scan labels under labels_root to find the maximum class id (>=0)."""
    max_c = -1
    for txt in labels_root.rglob("*.txt"):
        try:
            with open(txt, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    c = int(float(ln.split()[0]))
                    if c > max_c:
                        max_c = c
        except Exception:
            continue
    return max_c


def build_data_yaml_from_root(root: str,
                              train_subdir: str,
                              val_subdir: str,
                              names_list: Optional[List[str]],
                              names_file: Optional[str],
                              save_dir: Path) -> str:
    """Create a minimal data.yaml from a dataset root. Infer names if needed."""
    root_p = Path(root)
    if not root_p.exists():
        raise FileNotFoundError(f"--root path not found: {root}")
    # names priority: explicit list > names_file > names.txt in root > infer from labels
    if names_list and len(names_list) > 0:
        names = names_list
    elif names_file and Path(names_file).is_file():
        names = _read_names_file(Path(names_file))
    elif (root_p / "names.txt").is_file():
        names = _read_names_file(root_p / "names.txt")
    else:
        # infer from labels/train and labels/val
        train_labels = Path(root) / train_subdir.replace("images", "labels")
        val_labels = Path(root) / val_subdir.replace("images", "labels")
        max_c = max(_scan_max_class_id(train_labels),
                    _scan_max_class_id(val_labels))
        if max_c < 0:
            raise RuntimeError(
                "Could not infer classes: no label files found. Provide --names or --names-file.")
        names = [f"class{i}" for i in range(max_c + 1)]

    data = {
        "path": str(root_p),
        "train": train_subdir,
        "val": val_subdir,
        "names": names,
    }
    out_yaml = save_dir / "data_autogen.yaml"
    _write_yaml(data, out_yaml)
    return str(out_yaml)

# ------------------------------
# Balancing builder
# ------------------------------


def _gather_label_classes(label_path: Path) -> set:
    """Read a YOLO label file and return the set of class ids (ints)."""
    cs = set()
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                try:
                    c = int(float(parts[0]))
                    cs.add(c)
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return cs


def build_balanced_train_list(
    data_yaml_path: str,
    save_dir: Path,
    target_per_class: Optional[int] = None,
    balance_mult: Optional[float] = None,
) -> Tuple[str, dict]:
    """
    Build a balanced train image list for *multi-class* datasets.
    Strategy:
      - enumerate all train images
      - read each image's label file; get its set of classes
      - compute per-class freqs
      - set target_per_class = max(freq) if not given
      - sample images w/ prob proportional to sum(1/freq[c] for c in image_classes)
        until each class reaches target_per_class
    Returns (balanced_data_yaml_path:str, stats:dict)
    """
    cfg = _read_yaml(data_yaml_path)
    base = cfg.get("path", "")
    train_nodes = cfg.get("train")
    if train_nodes is None:
        raise ValueError("data.yaml missing 'train' key")

    image_roots = _resolve_to_list(base, train_nodes)
    images = _expand_image_dirs(image_roots)
    if not images:
        raise RuntimeError(
            "No training images found. Check your data.yaml 'path' and 'train'.")

    labels_roots = _derive_labels_roots(train_nodes, base)
    img_classes: List[set] = []
    class_freq: Dict[int, int] = {}
    for img in images:
        lbl = _image_to_label(img, image_roots, labels_roots)
        cs = _gather_label_classes(lbl)
        img_classes.append(cs)
        for c in cs:
            class_freq[c] = class_freq.get(c, 0) + 1

    if not class_freq:
        raise RuntimeError("No labeled objects found in training labels.")

    max_count = max(class_freq.values())
    if target_per_class is None:
        if balance_mult is not None:
            # Cap target as a fraction of the majority count
            target_per_class = max(1, int(balance_mult * max_count))
        else:
            target_per_class = max_count

    inv = {c: 1.0 / max(1, f) for c, f in class_freq.items()}
    weights = []
    for cs in img_classes:
        w = sum(inv.get(c, 0.0) for c in cs)
        weights.append(w if w > 0 else 1e-6)

    total_w = sum(weights)
    probs = [w / total_w for w in weights]

    per_class_added = {c: 0 for c in class_freq}
    picked: List[str] = []
    rng = random.Random(123)

    max_iters = len(images) * 30
    iters = 0
    while any(per_class_added[c] < target_per_class for c in per_class_added) and iters < max_iters:
        iters += 1
        i = rng.choices(range(len(images)), weights=probs, k=1)[0]
        cs = img_classes[i]
        if not cs:
            continue
        if any(per_class_added[c] < target_per_class for c in cs):
            picked.append(images[i])
            for c in cs:
                if per_class_added[c] < target_per_class:
                    per_class_added[c] += 1

    rng.shuffle(picked)

    # Write list and derived YAML
    list_path = save_dir / "balanced_train_multiclass.txt"
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(picked) + "\n", encoding="utf-8")

    out_yaml_path = save_dir / "data_balanced_multiclass.yaml"
    out_cfg = dict(cfg)
    out_cfg["train"] = str(list_path)
    _write_yaml(out_cfg, out_yaml_path)

    stats = {
        "n_images_total": len(images),
        "n_images_balanced_list": len(picked),
        "class_freq": class_freq,
        "target_per_class": target_per_class,
        "balance_mult": balance_mult,
        "labels_roots_used": labels_roots,
    }
    return str(out_yaml_path), stats

# ------------------------------
# Training
# ------------------------------


def train_yolo_seg(
    data_yaml: str,
    project: str,
    name: str,
    imgsz: int = 640,
    model_weights: Optional[str] = None,
    epochs: int = 40,
    batch: int = 32,
    seed: int = 42,
    use_balancing: bool = True,
    workers: Optional[int] = None,
    balance_target: Optional[int] = None,
    balance_mult: Optional[float] = None,
    lr0: float = 0.005,
    lrf: float = 0.05,
    retina_masks: bool = True,
    max_det: int = 300,
    fraction: float = 1.0,
    freeze: Optional[int] = None,
    rect: bool = True,
    resume: bool = False,
    auto_augment: str = "randaugment",
    erasing: float = 0.4,
    iou: float = 0.7,
    kobj: float = 1.0,
):
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build balanced train list (multi-class)
    data_for_train = data_yaml
    balance_stats = None
    if use_balancing:
        try:
            data_for_train, balance_stats = build_balanced_train_list(
                data_yaml, save_dir, target_per_class=balance_target, balance_mult=balance_mult)
            (save_dir / "balance_stats.json").write_text(json.dumps(balance_stats, indent=2))
            print("Balanced sampling enabled. Stats written to:",
                  save_dir / "balance_stats.json")
        except Exception as e:
            print("WARNING: Balancing failed, proceeding with original train split.\n", e)

    YOLO = _lazy_ultralytics()
    if model_weights is None:
        model_weights = "yolo11s-seg.pt"
    model = YOLO("yolo11s-seg.pt")

    # default workers: fewer on Windows to reduce DataLoader worker crashes
    if workers is None:
        workers = 6 if os.name == "nt" else 8

    train_overrides = dict(
        data=data_for_train,
        project=str(project),
        name=name,
        imgsz=imgsz,
        epochs=epochs,
        batch=batch,
        seed=seed,
        optimizer="AdamW",
        lr0=lr0,
        lrf=lrf,
        cos_lr=True,
        # imbalance helpers (removed fl_gamma: not supported in this Ultralytics version)
        copy_paste=0.0,      # stronger than default
        # good defaults for seg
        auto_augment=auto_augment,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        mosaic=0.0,
        close_mosaic=10,
        mixup=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=erasing,
        # masks
        retina_masks=retina_masks,
        rect=rect,
        max_det=max_det,
        iou=iou,
        kobj=kobj,
        # speed
        cache=False,
        fraction=fraction,
        freeze=freeze,
        workers=workers,
        # logs
        save_json=False,  # avoid faster-coco-eval dependency on Windows/Py39
        plots=False,     # save plots after val
        verbose=True,   # per-epoch results
        resume=resume,
    )

    print("\n[train] overrides:\n", json.dumps(train_overrides, indent=2))
    model.train(**train_overrides)

    # Validate and export per-class seg AP if available
    val_metrics = model.val(data=data_yaml, imgsz=imgsz, batch=batch, plots=True, 
                            rect=True, split="val", conf=0.05, iou=0.5, max_det=2000, retina_masks=True)
    try:
        seg_metrics = getattr(val_metrics, "seg", None) or getattr(
            getattr(val_metrics, "metrics", None), "seg", None)
        maps = getattr(seg_metrics, "maps", None)
        names = None
        try:
            names = model.model.names
        except Exception:
            names = None
        if maps is not None and names:
            per_class = {str(names[i]): float(maps[i])
                         for i in range(len(maps))}
            (save_dir / "seg_ap50_per_class.json").write_text(json.dumps(per_class, indent=2))
            print("Saved per-class seg AP50:",
                  save_dir / "seg_ap50_per_class.json")
    except Exception as e:
        print("Per-class seg AP export skipped:", e)

    print("\nDone. Runs dir:", save_dir)
# Predict and save sample outputs from validation set

# ------------------------------
# CLI
# ------------------------------


def _parse_args():
    ap = argparse.ArgumentParser(
        description="YOLO11-seg with multi-class balancing. Use --data or --root.")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--data", help="Path to data.yaml (YOLO format).")
    group.add_argument(
        "--root", help="Dataset root; expects images/train, labels/train, images/val, labels/val.")
    ap.add_argument("--project", required=True,
                    help="Project root to save runs.")
    ap.add_argument("--name", default="yolo11seg_balanced",
                    help="Run name (subfolder under project).")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="Training image size.")
    ap.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    ap.add_argument("--batch", type=int, default=16, help="Batch size.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--workers", type=int, default=None,
                    help="DataLoader workers (default 2 on Windows, 8 otherwise)")
    ap.add_argument("--model", default=None,
                    help="Model weights (e.g., yolo11m-seg.pt). Default yolo11n-seg.pt")
    ap.add_argument("--no-balance", action="store_true",
                    help="Disable multi-class balancing.")
    # training hyperparams
    ap.add_argument("--lr0", type=float, default=0.005,
                    help="Initial learning rate.")
    ap.add_argument("--lrf", type=float, default=0.05,
                    help="Final OneCycle LR factor.")
    ap.add_argument("--retina-masks", action="store_true",
                    help="Enable high-res masks.")
    ap.add_argument("--max-det", type=int, default=300,
                    help="Max detections per image.")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="Train on fraction of data.")
    ap.add_argument("--freeze", type=int, default=None,
                    help="Freeze first N layers.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume training (expects --model to point to last.pt or finds last run).")
    ap.add_argument("--auto-augment", default="randaugment",
                    help="Auto-augment policy (e.g., randaugment, ta_wide, None).")
    ap.add_argument("--erasing", type=float, default=0.4,
                    help="Random erasing probability.")
    ap.add_argument("--iou", type=float, default=0.7,
                    help="IoU threshold for NMS/evaluation.")
    ap.add_argument("--kobj", type=float, default=1.0,
                    help="Objectness loss gain.")

    # when using --root only:
    ap.add_argument("--train-subdir", default="images/train",
                    help="Relative train images directory under --root.")
    ap.add_argument("--val-subdir", default="images/val",
                    help="Relative val images directory under --root.")
    ap.add_argument("--names", default=None,
                    help="Comma-separated class names (overrides names-file).")
    ap.add_argument("--names-file", default=None,
                    help="Path to a text file with one class name per line.")
    ap.add_argument("--check-only", action="store_true",
                    help="Only run label sanity checks and exit.")
    ap.add_argument("--balance-target", type=int, default=None,
                    help="Absolute per-class target for balanced list (caps oversampling).")
    ap.add_argument("--balance-mult", type=float, default=None,
                    help="Target per-class as a fraction of majority count, e.g., 0.6.")
    return ap.parse_args()


def main():
    # If no CLI args given, use USER_CONFIG
    if len(os.sys.argv) <= 1:
        class _NS:
            pass
        cfg = USER_CONFIG
        args = _NS()
        # union of fields used below
        args.data = cfg.get("data")
        args.root = cfg.get("root") if cfg.get("use_root", True) else None
        args.project = cfg.get("project")
        args.name = cfg.get("name")
        args.imgsz = cfg.get("imgsz", 640)
        args.epochs = cfg.get("epochs", 40)
        args.batch = cfg.get("batch", 32)
        args.seed = cfg.get("seed", 42)
        args.workers = cfg.get("workers")
        args.model = cfg.get("model")
        args.no_balance = cfg.get("no_balance", False)
        args.train_subdir = cfg.get("train_subdir", "images/train")
        args.val_subdir = cfg.get("val_subdir", "images/val")
        args.names = cfg.get("names")
        args.names_file = cfg.get("names_file")
        args.check_only = False
        args.balance_target = cfg.get("balance_target")
        args.balance_mult = cfg.get("balance_mult")
        args.lr0 = cfg.get("lr0", 0.005)
        args.lrf = cfg.get("lrf", 0.05)
        args.retina_masks = cfg.get("retina_masks", True)
        args.max_det = cfg.get("max_det", 2000)
        args.fraction = cfg.get("fraction", 1.0)
        args.freeze = cfg.get("freeze")
        args.resume = cfg.get("resume", False)
        args.auto_augment = cfg.get("auto_augment", "randaugment")
        args.erasing = cfg.get("erasing", 0.4)
        args.iou = cfg.get("iou", 0.7)
        args.kobj = cfg.get("kobj", 1.0)
        if not args.project:
            raise SystemExit("Please set USER_CONFIG['project'].")
        if not (args.data or args.root):
            raise SystemExit(
                "Set USER_CONFIG['root'] or provide a data.yaml in USER_CONFIG['data'].")
    else:
        args = _parse_args()
    save_dir = Path(args.project) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build/resolve data.yaml
    if args.data:
        data_yaml_path = args.data
    else:
        names_list = [x.strip()
                      for x in args.names.split(",")] if args.names else None
        data_yaml_path = build_data_yaml_from_root(
            root=args.root,
            train_subdir=args.train_subdir,
            val_subdir=args.val_subdir,
            names_list=names_list,
            names_file=args.names_file,
            save_dir=save_dir,
        )
        print("Generated data.yaml:", data_yaml_path)

    if args.check_only:
        check_label_sanity(data_yaml_path, sample=500)
        return

    train_yolo_seg(
        data_yaml=data_yaml_path,
        project=args.project,
        name=args.name,
        imgsz=args.imgsz,
        model_weights=args.model,
        epochs=args.epochs,
        batch=args.batch,
        seed=args.seed,
        use_balancing=(not args.no_balance),
        workers=args.workers,
        balance_target=args.balance_target,
        balance_mult=args.balance_mult,
        lr0=args.lr0,
        lrf=args.lrf,
        retina_masks=args.retina_masks,
        max_det=args.max_det,
        fraction=args.fraction,
        freeze=args.freeze,
        resume=args.resume,
        auto_augment=args.auto_augment,
        erasing=args.erasing,
        iou=args.iou,
        kobj=args.kobj,
    )


if __name__ == "__main__":
    main()
