#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CLI prediction with calibrated softmax + confidence thresholds + calibration report.

Features
- ALWAYS assigns a label (argmax) to every image.
- Temperature scaling (fit once from a labeled validation split or load cached T).
- TTA (orig/hflip/vflip/hvflip) with averaged logits (optional).
- Confidence gating: global or per-class threshold; optional margin/entropy (for info).
- Outputs CSV with columns including passed_threshold, margin, entropy, topk.
- Writes meta.json recording all settings.
- Produces a calibration report (ECE/MCE/Brier) and reliability diagrams (pre- and post-calibration).

Example
-------
python predict_cli_calibrated_thresholds.py \
  --checkpoints-dir "E:/Santosh_master_thesis/Checkpoints_LOT_two_stages_10_species_640" \
  --labeled-data "E:/Santosh_master_thesis/LOT_flat_10_species" \
  --unlabeled-data "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data" \
  --out-csv "E:/Santosh_master_thesis/prediction_thresholded.csv" \
  --global-min 0.98 --per-class-min "Others=0.99" \
  --use-margin --margin-min 0.25

"""

import os, re, json, math, argparse, ast
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

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
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.transforms.functional as TF

# ----------------------------
# Defaults
# ----------------------------
DEFAULT_IMG_SIZE = 640
DEFAULT_TOPK = 3
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_SEED = 42
DEFAULT_GLOBAL_MIN = 0.98
DEFAULT_MARGIN_MIN = 0.25
DEFAULT_ENTROPY_MAX = 0.70
DEFAULT_N_BINS = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Transforms
# ----------------------------
def make_val_tf(image_size: int):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# ----------------------------
# Utilities
# ----------------------------
def parse_per_class_min(arg_str: Optional[str]) -> Dict[str, float]:
    """
    Parse "A=0.98,B=0.97" into {"A":0.98,"B":0.97}
    Accepts JSON dict too.
    """
    if not arg_str:
        return {}
    s = arg_str.strip()
    try:
        # Try JSON first
        d = json.loads(s)
        return {str(k): float(v) for k, v in d.items()}
    except Exception:
        pass
    # Fallback: comma-separated key=val
    out = {}
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" not in tok:
            raise ValueError(f"Invalid per-class threshold token: '{tok}' (expected key=val)")
        k, v = tok.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out

def find_best_checkpoint(checkpoint_dir: str) -> str:
    best_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("best_model_") and f.endswith(".pth")]
    if not best_files:
        raise FileNotFoundError("No best_model_*.pth in " + checkpoint_dir)
    pat = re.compile(r"best_model_(\d+)_([0-9]+\.[0-9]+)\.pth$")
    scored = []
    for f in best_files:
        m = pat.match(f)
        if m:
            scored.append((float(m.group(2)), f))
    if scored:
        scored.sort(key=lambda x: x[0])  # lowest val loss first
        return os.path.join(checkpoint_dir, scored[0][1])
    best_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)
    return os.path.join(checkpoint_dir, best_files[0])

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

def _collect_logits_labels(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
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

def load_or_fit_temperature(model, labeled_root, classes, out_dir,
                            cache_path: Optional[str], val_split=DEFAULT_VAL_SPLIT, seed=DEFAULT_SEED,
                            batch_size=32, image_size=DEFAULT_IMG_SIZE, n_bins=DEFAULT_N_BINS):
    # try load
    if cache_path and os.path.exists(cache_path):
        try:
            data = json.load(open(cache_path))
            return float(data["T"]), None  # no report if loading cached
        except Exception:
            pass

    # build val split from labeled_root
    base = datasets.ImageFolder(labeled_root)
    y = [lbl for _, lbl in base.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=seed)
    _, val_idx = next(sss.split(np.zeros(len(y)), y))
    val_set = datasets.ImageFolder(labeled_root, transform=make_val_tf(image_size))
    val_set.samples = [base.samples[i] for i in val_idx]
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # collect logits (pre-calibration)
    logits, labels = _collect_logits_labels(model, val_loader)

    # report pre-calibration
    report_pre = calibration_report_from_logits(logits, labels, n_bins=n_bins)

    # fit T
    T = _fit_temperature_on_logits(logits, labels)

    # report post-calibration
    logits_post = logits / T
    report_post = calibration_report_from_logits(logits_post, labels, n_bins=n_bins)

    # save temperature
    payload = {"T": T, "report_pre": report_pre, "report_post": report_post}
    if cache_path:
        try:
            json.dump(payload, open(cache_path, "w"), indent=2)
        except Exception:
            pass

    # save reliability diagrams
    os.makedirs(out_dir, exist_ok=True)
    save_reliability_diagram(report_pre, os.path.join(out_dir, "reliability_pre.png"))
    save_reliability_diagram(report_post, os.path.join(out_dir, "reliability_post.png"))
    json.dump(report_pre, open(os.path.join(out_dir, "calibration_pre.json"), "w"), indent=2)
    json.dump(report_post, open(os.path.join(out_dir, "calibration_post.json"), "w"), indent=2)

    return T, {"pre": report_pre, "post": report_post}

# ----------------------------
# Calibration metrics & reliability diagram
# ----------------------------
def calibration_report_from_logits(logits: torch.Tensor, labels: torch.Tensor, n_bins=DEFAULT_N_BINS):
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)
        correct = preds.eq(labels).float()

        ece, mce, bins = compute_ece_mce(conf.cpu().numpy(), correct.cpu().numpy(), n_bins=n_bins)
        brier = compute_brier(probs.cpu().numpy(), labels.cpu().numpy())

        return {
            "ece": float(ece),
            "mce": float(mce),
            "brier": float(brier),
            "bins": bins,  # list of dicts with bin stats
        }

def compute_ece_mce(confidences: np.ndarray, correct: np.ndarray, n_bins=DEFAULT_N_BINS):
    bins = []
    ece = 0.0
    mce = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i+1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        idx = np.where(mask)[0]
        if idx.size == 0:
            bins.append({"bin": [float(lo), float(hi)], "n": 0, "acc": None, "conf": None, "gap": None})
            continue
        bin_conf = float(confidences[idx].mean())
        bin_acc = float(correct[idx].mean())
        gap = abs(bin_acc - bin_conf)
        w = idx.size / confidences.size
        ece += w * gap
        mce = max(mce, gap)
        bins.append({"bin": [float(lo), float(hi)], "n": int(idx.size), "acc": bin_acc, "conf": bin_conf, "gap": gap})
    return ece, mce, bins

def compute_brier(probs: np.ndarray, labels: np.ndarray):
    # multi-class Brier = mean over samples of sum_k (p_k - y_k)^2
    n = probs.shape[0]
    k = probs.shape[1]
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), labels] = 1.0
    sq = (probs - onehot) ** 2
    return float(sq.sum(axis=1).mean())

def save_reliability_diagram(report: dict, out_path: str):
    # bar chart of accuracy per bin with line for confidence
    bins = report["bins"]
    xs, accs, confs, ns = [], [], [], []
    for b in bins:
        lo, hi = b["bin"]
        xs.append((lo + hi) / 2.0)
        accs.append(b["acc"] if b["acc"] is not None else np.nan)
        confs.append(b["conf"] if b["conf"] is not None else np.nan)
        ns.append(b["n"])

    plt.figure(figsize=(7,5))
    width = (1.0 / len(xs)) * 0.9
    plt.bar(xs, accs, width=width, alpha=0.6, label="Accuracy per bin")
    plt.plot(xs, confs, marker="o", label="Avg confidence")
    plt.plot([0,1],[0,1], "k--", label="Perfect calibration")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy / Confidence")
    plt.title(f"Reliability Diagram (ECE={report['ece']:.3f}, Brier={report['brier']:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ----------------------------
# TTA
# ----------------------------
def tta_views(pil_img: Image.Image, mode: str) -> List[Image.Image]:
    if mode == "none":
        return [pil_img]
    # default: hvflip4
    return [pil_img,
            TF.hflip(pil_img),
            TF.vflip(pil_img),
            TF.vflip(TF.hflip(pil_img))]

# ----------------------------
# Prediction
# ----------------------------
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
    ap = argparse.ArgumentParser(description="Predict with calibrated softmax + thresholds + calibration report")
    ap.add_argument("--checkpoints-dir", type=str, required=True, help="Folder containing best_model_*.pth")
    ap.add_argument("--checkpoint", type=str, default=None, help="Optional explicit checkpoint .pth path")
    ap.add_argument("--labeled-data", type=str, required=True, help="Labeled dataset (ImageFolder) for class order and calibration split")
    ap.add_argument("--unlabeled-data", type=str, required=True, help="Unlabeled root with subfolders")
    ap.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    ap.add_argument("--out-dir", type=str, default=None, help="Directory to write calibration artifacts (default: alongside CSV)")

    ap.add_argument("--image-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--tta", type=str, default="hvflip4", choices=["hvflip4","none"])

    # Temperature calibration
    ap.add_argument("--no-temperature", action="store_true", help="Disable temperature scaling")
    ap.add_argument("--temp-cache", type=str, default=None, help="Path to cache the learned temperature (json)")
    ap.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT, help="Val split ratio for calibration")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--calib-batch-size", type=int, default=32)
    ap.add_argument("--calib-bins", type=int, default=DEFAULT_N_BINS)

    # Thresholding (we still label everything)
    ap.add_argument("--global-min", type=float, default=DEFAULT_GLOBAL_MIN, help="Global min top-1 prob to pass")
    ap.add_argument("--per-class-min", type=str, default=None, help='Per-class thresholds, e.g. \'{"Others":0.99}\' or "Leaves=0.98,Trunks=0.98"')
    ap.add_argument("--use-margin", action="store_true")
    ap.add_argument("--margin-min", type=float, default=DEFAULT_MARGIN_MIN)
    ap.add_argument("--use-entropy", action="store_true")
    ap.add_argument("--entropy-max", type=float, default=DEFAULT_ENTROPY_MAX)

    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(args.out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)

    # Load labeled set for classes
    val_tf = make_val_tf(args.image_size)
    labeled_ds = datasets.ImageFolder(args.labeled_data, transform=val_tf)
    class_names = labeled_ds.classes
    num_classes = len(class_names)

    # Load model
    if args.checkpoint:
        ckpt = args.checkpoint
    else:
        ckpt = find_best_checkpoint(args.checkpoints_dir)

    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_features, num_classes))
    state = torch.load(ckpt, map_significant_value_to_device := DEVICE)  # noqa: E251 (pep8)
    # Python <3.8 compatible: split the above line to avoid walrus operator confusion
    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval().to(DEVICE)

    # Temperature
    if args.no_temperature:
        T = 1.0
        calib_report = None
    else:
        T, calib_report = load_or_fit_temperature(
            model, args.labeled_data, class_names, out_dir,
            cache_path=args.temp_cache, val_split=args.val_split,
            seed=args.seed, batch_size=args.calib_batch_size,
            image_size=args.image_size, n_bins=args.calib_bins
        )

    # Parse thresholds
    per_class_min = parse_per_class_min(args.per_class_min)

    # Predict unlabeled
    rows = []
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    for sub in os.listdir(args.unlabeled_data):
        subdir = os.path.join(args.unlabeled_data, sub)
        if not os.path.isdir(subdir):
            continue
        files = [f for f in os.listdir(subdir) if f.lower().endswith(image_exts)]
        for fname in tqdm(files, desc=f"Predicting {sub}"):
            fpath = os.path.join(subdir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
            except (OSError, UnidentifiedImageError):
                rows.append({"image_path": fpath, "error": "unreadable"})
                continue

            try:
                probs, top_classes, top_probs, p1, p2, margin, ent = predict_one(
                    model, img, val_tf, DEVICE, T, args.tta, args.topk
                )
                pred_idx = int(top_classes[0])
                pred_name = class_names[pred_idx]
                thr = per_class_min.get(pred_name, args.global_min)

                passed = (p1 >= thr)
                if args.use_margin:
                    passed = passed and (margin >= args.margin_min)
                if args.use_entropy:
                    passed = passed and (ent <= args.entropy_max)

                rows.append({
                    "image_path": fpath,
                    "predicted_class": pred_name,
                    "confidence": round(float(p1), 6),
                    "threshold_used": round(float(thr), 6),
                    "passed_threshold": bool(passed),
                    "margin_to_2nd": round(float(margin), 6),
                    "entropy": round(float(ent), 6),
                    "topk_classes": [class_names[i] for i in top_classes],
                    "topk_probs": [round(float(p), 6) for p in top_probs],
                })
            except Exception as e:
                rows.append({"image_path": fpath, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    # Meta
    meta = {
        "checkpoint": ckpt,
        "class_order": class_names,
        "tta": args.tta,
        "temperature": T,
        "used_temperature": not args.no_temperature,
        "global_min_conf": args.global_min,
        "per_class_min": per_class_min,
        "use_margin": args.use_margin,
        "margin_min": args.margin_min,
        "use_entropy": args.use_entropy,
        "entropy_max": args.entropy_max,
        "topk": args.topk,
        "out_dir": out_dir,
        "calibration": calib_report if calib_report is not None else "skipped",
    }
    with open(os.path.join(out_dir, os.path.basename(args.out_csv).replace(".csv","_meta.json")), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Saved predictions to {args.out_csv}")
    if not args.no_temperature:
        print(f"Calibration artifacts: {os.path.join(out_dir, 'reliability_pre.png')} and 'reliability_post.png'")
        print(f"Calibration JSON: {os.path.join(out_dir, 'calibration_pre.json')} / 'calibration_post.json'")

if __name__ == "__main__":
    main()
