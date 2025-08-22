import argparse
import os
import re
import json
import datetime as dt
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score, accuracy_score


def parse_args():
    p = argparse.ArgumentParser(
        description="Screen checkpoints to find best by F1-macro and Accuracy. Uses existing stats if available, else can evaluate on a labeled dataset.")
    p.add_argument("--roots", nargs="+", default=[
                   "E:/Santosh_master_thesis"], help="Directories to scan recursively for .pth")
    p.add_argument("--classes", type=int, default=None,
                   help="Filter by expected number of classes (e.g., 9). If omitted, accept any.")
    p.add_argument("--name-hint", default="",
                   help="Substring to filter checkpoint paths (case-insensitive). Empty = no filter.")
    p.add_argument("--from", dest="from_date", default="2025-06-01",
                   help="Start date YYYY-MM-DD (inclusive). Optional")
    p.add_argument("--to", dest="to_date", default="2025-07-31",
                   help="End date YYYY-MM-DD (inclusive). Optional")
    p.add_argument("--labeled-data", dest="labeled_data", default=None,
                   help="If set, evaluate checkpoints on this ImageFolder dataset (val split 20%).")
    p.add_argument("--image-size", type=int, default=512,
                   help="Image size for eval transforms")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for eval")
    p.add_argument("--limit", type=int, default=200,
                   help="Max checkpoints to inspect")
    p.add_argument("--topk", type=int, default=10,
                   help="Print top-K models by F1 macro")
    p.add_argument("--export-csv", default=None,
                   help="Optional path to save a CSV summary")
    p.add_argument("--export-confusions", default=None,
                   help="Optional directory to save confusion matrices for top-K (requires --labeled-data)")
    return p.parse_args()


def parse_date(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    return dt.datetime.fromisoformat(s)


def within_range(path: str, start: Optional[dt.datetime], end: Optional[dt.datetime]) -> bool:
    try:
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return False
    if start and mtime < start:
        return False
    if end:
        # inclusive end of day
        end2 = end + dt.timedelta(days=1) - dt.timedelta(seconds=1)
        if mtime > end2:
            return False
    return True


def locate_stats_dir(cp_path: str) -> Optional[str]:
    d = os.path.dirname(cp_path)
    for cand in [os.path.join(d, "Training_Stats"), os.path.join(os.path.dirname(d), "Training_Stats")]:
        if os.path.isdir(cand):
            return cand
    return None


def read_existing_stats(cp_path: str) -> Optional[Dict]:
    stats_dir = locate_stats_dir(cp_path)
    if not stats_dir:
        return None
    report = os.path.join(stats_dir, "classification_report.json")
    if not os.path.isfile(report):
        return None
    try:
        with open(report, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def infer_num_classes_from_report(report: Dict) -> Optional[int]:
    labels = [k for k in report.keys() if k not in (
        "accuracy", "macro avg", "weighted avg")]
    try:
        return len(labels)
    except Exception:
        return None


def eval_checkpoint(cp_path: str, data_dir: str, image_size: int, batch_size: int, device: torch.device) -> Tuple[float, float, float, np.ndarray, List[str]]:
    base = datasets.ImageFolder(data_dir)
    class_names = base.classes
    targets = [s[1] for s in base.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    val_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_set = datasets.ImageFolder(data_dir, transform=val_tf)
    val_set.samples = [base.samples[i] for i in val_idx]
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    num_classes = len(class_names)
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, num_classes))

    state = torch.load(cp_path, map_location=device)
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        # try common wrapper
        if isinstance(state, dict) and "state_dict" in state:
            try:
                model.load_state_dict(state["state_dict"], strict=False)
            except Exception:
                pass

    model.eval().to(device)

    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(preds.detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    balc = balanced_accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return acc, f1m, balc, cm, class_names


def screen_checkpoints(args):
    name_hint = args.name_hint.lower() if args.name_hint else ""
    start = parse_date(args.from_date)
    end = parse_date(args.to_date)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Gather candidates
    candidates = []
    for root in args.roots:
        for dirpath, _, filenames in os.walk(root):
            if name_hint and name_hint not in dirpath.lower():
                continue
            for fn in filenames:
                if not fn.lower().endswith(".pth"):
                    continue
                p = os.path.join(dirpath, fn)
                if not within_range(p, start, end):
                    continue
                candidates.append(p)

    # Limit
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    candidates = candidates[: args.limit]

    rows = []
    for cp in candidates:
        stats = read_existing_stats(cp)
        acc = f1m = balc = None
        n_classes = None
        stats_source = "none"

        if stats:
            stats_source = "json"
            try:
                acc = float(stats.get("accuracy", None))
            except Exception:
                acc = None
            try:
                f1m = float(stats.get("macro avg", {}).get("f1-score", None))
            except Exception:
                f1m = None
            # infer class count
            n_classes = infer_num_classes_from_report(stats)

        # If no stats or evaluation requested, evaluate
        if args.labeled_data and (acc is None or f1m is None):
            try:
                acc, f1m, balc, cm, class_names = eval_checkpoint(
                    cp, args.labeled_data, args.image_size, args.batch_size, device)
                stats_source = "eval"
            except Exception:
                pass

        if args.classes is not None and n_classes is not None and n_classes != args.classes:
            continue

        mtime = dt.datetime.fromtimestamp(
            os.path.getmtime(cp)).strftime("%Y-%m-%d %H:%M:%S")
        rows.append({
            "path": cp,
            "mtime": mtime,
            "classes": n_classes,
            "accuracy": acc,
            "f1_macro": f1m,
            "balanced_acc": balc,
            "stats_source": stats_source,
        })

    # Rank by f1 then accuracy (descending)
    def sort_key(r):
        return (
            -1.0 if r["f1_macro"] is None else -r["f1_macro"],
            -1.0 if r["accuracy"] is None else -r["accuracy"],
        )

    rows.sort(key=sort_key)

    # Print top-K
    topk = rows[: args.topk]
    print(f"\nTop {len(topk)} models by F1-macro, then Accuracy:")
    for i, r in enumerate(topk, 1):
        print(f"[{i}] f1={r['f1_macro'] if r['f1_macro'] is not None else 'n/a':>6} acc={r['accuracy'] if r['accuracy'] is not None else 'n/a':>6} src={r['stats_source']:<4} classes={r['classes'] if r['classes'] is not None else 'n/a'}")
        print(f"     {r['path']}")

    # Optional CSV export
    if args.export_csv:
        import csv
        with open(args.export_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                               "path", "mtime", "classes", "accuracy", "f1_macro", "balanced_acc", "stats_source"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Saved summary CSV to {args.export_csv}")

    # Optional confusion export for top-K (only if we evaluated)
    if args.export_confusions and args.labeled_data:
        os.makedirs(args.export_confusions, exist_ok=True)
        exported = 0
        for r in topk:
            if r["stats_source"] != "eval":
                continue
            try:
                acc, f1m, balc, cm, class_names = eval_checkpoint(
                    r["path"], args.labeled_data, args.image_size, args.batch_size, device)
            except Exception:
                continue
            # Save as CSV
            base = os.path.splitext(os.path.basename(r["path"]))[0]
            out_csv = os.path.join(args.export_confusions, f"{base}_cm.csv")
            np.savetxt(out_csv, cm, fmt="%d", delimiter=",")
            exported += 1
        print(
            f"Exported {exported} confusion matrices to {args.export_confusions}")


if __name__ == "__main__":
    args = parse_args()
    screen_checkpoints(args)
