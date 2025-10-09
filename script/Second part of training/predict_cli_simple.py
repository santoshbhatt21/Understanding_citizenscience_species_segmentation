#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Label images with EfficientNet V2-S and export CSV + per-class summary.

- CLI exactly as requested via `_parse_args()` (checkpoints dir, labeled data, unlabeled data, output csv, image size, TTA mode)
- ALWAYS labels every image (argmax), writes calibrated softmax confidence, margin to 2nd, entropy, and top-k.
- Uses Temperature Scaling (fits once from a small validation split of the labeled set and caches T next to the CSV).
- Supports simple TTA: none | flip4 (orig, hflip, vflip, hvflip).
- Recursively scans the unlabeled root for images.
- NEW: emits a per-class summary (count, mean confidence/margin/entropy, pass rate when thresholds enabled)
        printed to console and saved as *_summary.json and *_summary.csv.

Edit these constants below if you also want a threshold flag in the CSV:
    GLOBAL_MIN = 0.98
    PER_CLASS_MIN = {"Others": 0.99}  # example
Set GLOBAL_MIN=None to disable the passed_threshold column.
"""

import os
import re
import json
import math
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.functional as TF
from tqdm import tqdm

# ----------------------------
# Constants you can edit
# ----------------------------
TOPK = 3
VAL_SPLIT = 0.2
SEED = 42

# Optional threshold flagging (edit to taste)
GLOBAL_MIN: Optional[float] = 0.98  # set to None to disable
PER_CLASS_MIN = {
    # "Others": 0.99,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# CLI (as requested)
# ----------------------------


def _parse_args():
    p = argparse.ArgumentParser(
        description="Label images with EfficientNet V2-S and export CSV.")
    p.add_argument("--checkpoints-dir", default="E:/Santosh_master_thesis/Ckpt_efficientnet_v2s_rootLOSS_1500_images",
                   help="Directory containing best_model_*.pth checkpoints")
    p.add_argument("--labeled-data-path", default="E:/Santosh_master_thesis/flat_labeled_Leaves_Others_Trunks_1500_images",
                   help="Folder with class subfolders used to derive class order")
    p.add_argument("--unlabeled-data-path", default="E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data",
                   help="Root folder to scan for images (recursively)")
    p.add_argument("--output-csv-path", default="E:/Santosh_master_thesis/prediction_metadata_Leaves_Others_Trunks.csv",
                   help="Where to save the predictions CSV")
    p.add_argument("--image-size", type=int, default=640,
                   help="Inference image size (center-crop)")
    p.add_argument("--tta", choices=["none", "flip4"],
                   default="flip4", help="Test-time augmentation policy")
    return p.parse_args()

# ----------------------------
# Transforms
# ----------------------------


def make_val_tf(image_size: int):
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ----------------------------
# Utilities
# ----------------------------


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """Pick best_model_*.pth with lowest loss in filename; fallback to most recent."""
    best_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(
        "best_model_") and f.endswith(".pth")]
    if not best_files:
        raise FileNotFoundError(
            f"No best_model_*.pth found in {checkpoint_dir}")
    pat = re.compile(r"best_model_(\d+)_([0-9]+\.[0-9]+)\.pth$")
    scored = []
    for f in best_files:
        m = pat.match(f)
        if m:
            scored.append((float(m.group(2)), f))
    if scored:
        scored.sort(key=lambda x: x[0])  # lowest val loss first
        return os.path.join(checkpoint_dir, scored[0][1])
    best_files.sort(key=lambda f: os.path.getmtime(
        os.path.join(checkpoint_dir, f)), reverse=True)
    return os.path.join(checkpoint_dir, best_files[0])


def walk_images(root: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    paths = []
    for d, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(exts):
                paths.append(os.path.join(d, fn))
    return paths


def entropy_from_probs(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())

# ----------------------------
# Temperature scaling (single scalar T)
# ----------------------------


class _TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_T = nn.Parameter(torch.zeros(1))  # T=exp(log_T) ~ 1

    def forward(self, logits):
        T = torch.exp(self.log_T)
        return logits / T


@torch.no_grad()
def _collect_logits_labels(model, loader):
    model.eval()
    logits_list, labels_list = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        z = model(x)
        logits_list.append(z)
        labels_list.append(y)
    return torch.cat(logits_list, 0), torch.cat(labels_list, 0)


def _fit_temperature_on_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    nll = nn.CrossEntropyLoss(reduction="mean")
    scaler = _TempScaler().to(DEVICE)
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = nll(scaler(logits), labels)
        loss.backward()
        return loss

    _ = opt.step(closure)
    T = float(torch.exp(scaler.log_T).item())
    return T


def load_or_fit_temperature(model, labeled_root, image_size, cache_path: str):
    # Try load
    if cache_path and os.path.exists(cache_path):
        try:
            data = json.load(open(cache_path, "r"))
            return float(data["T"]), data.get("report_pre"), data.get("report_post")
        except Exception:
            pass
    # Build a small val split from labeled set
    base = datasets.ImageFolder(labeled_root)
    y = [lbl for _, lbl in base.samples]
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SPLIT, random_state=SEED)
    _, val_idx = next(sss.split(np.zeros(len(y)), y))
    val_set = datasets.ImageFolder(
        labeled_root, transform=make_val_tf(image_size))
    val_set.samples = [base.samples[i] for i in val_idx]
    val_loader = DataLoader(val_set, batch_size=32,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Collect logits and fit T
    logits, labels = _collect_logits_labels(model, val_loader)
    T = _fit_temperature_on_logits(logits, labels)

    # Save
    payload = {"T": T}
    if cache_path:
        try:
            json.dump(payload, open(cache_path, "w"), indent=2)
        except Exception:
            pass
    return T, None, None

# ----------------------------
# TTA
# ----------------------------


def tta_views(pil_img: Image.Image, mode: str) -> List[Image.Image]:
    if mode == "none":
        return [pil_img]
    return [pil_img,
            TF.hflip(pil_img),
            TF.vflip(pil_img),
            TF.vflip(TF.hflip(pil_img))]


@torch.no_grad()
def predict_one(model, pil_img, tf, device, T: float, tta_mode: str, topk: int):
    imgs = tta_views(pil_img, tta_mode)
    logits_sum = None
    for im in imgs:
        x = tf(im).unsqueeze(0).to(device, non_blocking=True)
        z = model(x)
        logits_sum = z if logits_sum is None else (logits_sum + z)
    logits = logits_sum / len(imgs)
    logits = logits / T  # temperature BEFORE softmax
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    idx = probs.argsort()[::-1][:topk]
    top_classes = idx.tolist()
    top_probs = probs[idx].tolist()

    p1 = float(top_probs[0])
    p2 = float(top_probs[1]) if len(top_probs) > 1 else 0.0
    margin = p1 - p2
    ent = entropy_from_probs(probs)

    return probs, top_classes, top_probs, p1, p2, margin, ent

# ----------------------------
# Main
# ----------------------------


def main():
    args = _parse_args()

    # Load class order from labeled folder
    val_tf = make_val_tf(args.image_size)
    labeled_ds = datasets.ImageFolder(args.labeled_data_path, transform=val_tf)
    class_names = labeled_ds.classes
    num_classes = len(class_names)

    # Load checkpoint & model
    ckpt = find_best_checkpoint(args.checkpoints_dir)
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, num_classes))
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    # Temperature (cache next to CSV)
    temp_cache = os.path.splitext(args.output_csv_path)[
        0] + "_temperature.json"
    T, _, _ = load_or_fit_temperature(
        model, args.labeled_data_path, args.image_size, cache_path=temp_cache)
    print(f"[Info] Using temperature T = {T:.3f} (cache: {temp_cache})")

    # Walk unlabeled images recursively
    img_paths = walk_images(args.unlabeled_data_path)
    print(
        f"[Info] Found {len(img_paths)} images under: {args.unlabeled_data_path}")

    rows = []
    for fpath in tqdm(img_paths, desc="Predicting"):
        try:
            img = Image.open(fpath).convert("RGB")
        except (OSError, UnidentifiedImageError):
            rows.append({"image_path": fpath, "error": "unreadable"})
            continue

        try:
            probs, top_classes, top_probs, p1, p2, margin, ent = predict_one(
                model, img, val_tf, DEVICE, T, args.tta, TOPK
            )
            pred_idx = int(top_classes[0])
            pred_name = class_names[pred_idx]

            row = {
                "image_path": fpath,
                "predicted_class": pred_name,
                "confidence": round(float(p1), 6),
                "margin_to_2nd": round(float(margin), 6),
                "entropy": round(float(ent), 6),
                "topk_classes": [class_names[i] for i in top_classes],
                "topk_probs": [round(float(p), 6) for p in top_probs],
                "checkpoint": os.path.basename(ckpt),
                "tta": args.tta,
                "temperature": round(float(T), 6),
            }

            # Optional threshold flag
            if GLOBAL_MIN is not None:
                thr = PER_CLASS_MIN.get(pred_name, GLOBAL_MIN)
                row["threshold_used"] = round(float(thr), 6)
                row["passed_threshold"] = bool(p1 >= thr)

            rows.append(row)
        except Exception as e:
            rows.append({"image_path": fpath, "error": str(e)})

    # Save CSV
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv_path) or ".", exist_ok=True)
    df.to_csv(args.output_csv_path, index=False)
    print(f"\n✅ Saved predictions to: {args.output_csv_path}")
    print(f"   Classes: {class_names}")
    print(f"   Checkpoint: {ckpt}")

    # Save meta
    meta = {
        "checkpoint": ckpt,
        "class_order": class_names,
        "tta": args.tta,
        "temperature": T,
        "image_size": args.image_size,
        "thresholds": {"global": GLOBAL_MIN, "per_class": PER_CLASS_MIN} if GLOBAL_MIN is not None else "disabled",
        "num_images": len(img_paths),
    }
    meta_path = os.path.splitext(args.output_csv_path)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"   Meta saved to: {meta_path}")

    # ----------------------------
    # NEW: Per-class summary
    # ----------------------------
    print("\n--- Per-class summary ---")
    has_pred = df["predicted_class"].notna(
    ) if "predicted_class" in df.columns else pd.Series([], dtype=bool)
    df_valid = df[has_pred].copy(
    ) if "predicted_class" in df.columns else pd.DataFrame()

    summary_rows = []
    classes_in_pred = sorted(
        df_valid["predicted_class"].unique()) if not df_valid.empty else []
    has_thresh = ("passed_threshold" in df_valid.columns)
    if has_thresh and not df_valid.empty:
        # Ensure boolean dtype to avoid bitwise/int surprises
        try:
            df_valid["passed_threshold"] = df_valid["passed_threshold"].astype(
                bool)
        except Exception:
            pass

    for cls in class_names:  # include classes even if count=0
        sub = df_valid[df_valid["predicted_class"] == cls]
        n = int(len(sub))
        mean_conf = float(sub["confidence"].mean()) if n > 0 else None
        mean_margin = float(sub["margin_to_2nd"].mean()) if n > 0 else None
        mean_entropy = float(sub["entropy"].mean()) if n > 0 else None
        if has_thresh and n > 0:
            n_pass = int(sub["passed_threshold"].astype(bool).sum())
            n_fail = int(n - n_pass)
            pass_rate = (n_pass / n) if n > 0 else None
            thr_used = float(sub["threshold_used"].astype(
                float).mean()) if "threshold_used" in sub.columns else None
        else:
            pass_rate = None
            n_pass = None
            n_fail = None
            thr_used = None

        summary_rows.append({
            "class": cls,
            "n": n,
            "mean_confidence": round(mean_conf, 6) if mean_conf is not None else None,
            "mean_margin": round(mean_margin, 6) if mean_margin is not None else None,
            "mean_entropy": round(mean_entropy, 6) if mean_entropy is not None else None,
            "threshold_used": round(thr_used, 6) if thr_used is not None else None,
            "n_pass": n_pass,
            "n_fail": n_fail,
            "pass_rate": round(pass_rate, 6) if pass_rate is not None else None,
        })

    # Overall stats
    total_rows = int(len(df_valid))
    overall = {
        "class": "_OVERALL_",
        "n": total_rows,
        "mean_confidence": round(float(df_valid["confidence"].mean()), 6) if total_rows > 0 else None,
        "mean_margin": round(float(df_valid["margin_to_2nd"].mean()), 6) if total_rows > 0 else None,
        "mean_entropy": round(float(df_valid["entropy"].mean()), 6) if total_rows > 0 else None,
    }
    if has_thresh and total_rows > 0:
        n_pass_all = int(df_valid["passed_threshold"].astype(bool).sum())
        n_fail_all = int(total_rows - n_pass_all)
        overall["n_pass"] = n_pass_all
        overall["n_fail"] = n_fail_all
        overall["pass_rate"] = round(float(n_pass_all / total_rows), 6)

    summary_rows.append(overall)

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    # Save summary next to CSV
    base = os.path.splitext(args.output_csv_path)[0]
    summary_json_path = base + "_summary.json"
    summary_csv_path = base + "_summary.csv"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    summary_df.to_csv(summary_csv_path, index=False)
    print(
        f"\n📝 Summary saved to:\n  - {summary_json_path}\n  - {summary_csv_path}")


if __name__ == "__main__":
    main()
