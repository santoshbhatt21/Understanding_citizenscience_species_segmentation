import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(
        description="Find species checkpoint(s) whose confusion matrix matches a target CSV.")
    p.add_argument("--dirs", nargs="+", required=True,
                   help="Directories to scan recursively for .pth checkpoints")
    p.add_argument("--data", required=True,
                   help="ImageFolder root used for evaluation (species dataset; class order must match target)")
    p.add_argument("--target-cm", required=True,
                   help="Path to target confusion matrix CSV (NxN numbers; class order must match dataset)")
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=999999,
                   help="Max checkpoints to evaluate")
    p.add_argument("--tolerance", type=float, default=0.0,
                   help="Allowed absolute per-cell difference for matching (0 = exact)")
    p.add_argument("--normalized", action="store_true",
                   help="Compare row-normalized matrices with float tolerance instead of raw counts")
    p.add_argument("--save-matches", default=None,
                   help="If set, save each matched CM as CSV into this directory")
    return p.parse_args()


def load_target_cm(path: str) -> np.ndarray:
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Target CM must be square; got shape {arr.shape}")
    return arr


def row_normalize(cm: np.ndarray) -> np.ndarray:
    m = cm.astype(float)
    row_sums = m.sum(axis=1, keepdims=True)
    with np.errstate(all='ignore'):
        m = np.divide(m, row_sums, out=np.zeros_like(m), where=row_sums != 0)
    return m


def cm_match(a: np.ndarray, b: np.ndarray, tol: float, normalized: bool) -> bool:
    if a.shape != b.shape:
        return False
    if normalized:
        a = row_normalize(a)
        b = row_normalize(b)
    return np.all(np.abs(a - b) <= tol)


def eval_cm_for_checkpoint(cp_path: str, data_dir: str, image_size: int, batch_size: int, device: torch.device) -> np.ndarray:
    base = datasets.ImageFolder(data_dir)
    class_names = base.classes
    targets = [s[1] for s in base.samples]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(sss.split(np.zeros(len(targets)), targets))

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

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    return cm


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target = load_target_cm(args.target_cm)

    # Collect checkpoints
    cps: List[str] = []
    for root in args.dirs:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith('.pth'):
                    cps.append(os.path.join(dirpath, fn))
    cps.sort(key=lambda p: os.path.getmtime(p))

    print(f"Evaluating {min(len(cps), args.limit)} of {len(cps)} checkpoints…")
    if args.save_matches:
        os.makedirs(args.save_matches, exist_ok=True)

    matched = []
    for i, cp in enumerate(cps[: args.limit], 1):
        print(f"[{i}/{min(len(cps), args.limit)}] {cp}")
        try:
            cm = eval_cm_for_checkpoint(
                cp, args.data, args.image_size, args.batch_size, device)
        except Exception as e:
            print(f"  Skip due to error: {e}")
            continue

        if cm.shape != target.shape:
            print(f"  Shape mismatch: got {cm.shape}, want {target.shape}")
            continue

        if cm_match(cm.astype(float), target.astype(float), tol=args.tolerance, normalized=args.normalized):
            print("  MATCH!")
            matched.append(cp)
            if args.save_matches:
                base = os.path.splitext(os.path.basename(cp))[0]
                out_csv = os.path.join(args.save_matches, f"{base}_cm.csv")
                np.savetxt(out_csv, cm, fmt="%d", delimiter=",")

    if matched:
        print("\nMatched checkpoints:")
        for p in matched:
            print(p)
    else:
        print("\nNo matches found. Consider using --normalized and a non-zero --tolerance, and ensure class order matches.")


if __name__ == "__main__":
    main()
