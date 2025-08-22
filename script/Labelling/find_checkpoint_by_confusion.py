import os
import argparse
from typing import Optional, Dict, List

import numpy as np
import torch
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt


def plot_confusion_matrix_png(cm: np.ndarray, labels: List[str], out_path: str, normalize: bool = False, title: Optional[str] = None):
    plt.figure(figsize=(8, 6))
    matrix = cm.astype(float)
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(
                matrix), where=row_sums != 0)
    im = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha='right')
    plt.yticks(ticks, labels)
    fmt = ".2f" if normalize else ".0f"
    thresh = (matrix.max() if matrix.size else 0) / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            plt.text(j, i, format(val, fmt), ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def build_val_loader(data_path: str, image_size: int, seed: int, batch_size: int):
    base = datasets.ImageFolder(data_path, transform=None)
    targets = [s[1] for s in base.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    _, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    val_tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_set = datasets.ImageFolder(data_path, transform=val_tfm)
    val_set.samples = [base.samples[i] for i in val_idx]
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    return val_loader, val_set.classes


def load_model(num_classes: int):
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(
        0.5), torch.nn.Linear(in_features, num_classes))
    return model


def evaluate_checkpoint(ckpt_path: str, val_loader, num_classes: int, device: str):
    model = load_model(num_classes)
    try:
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
    except Exception as e:
        print(f"Skip {ckpt_path}: state_dict mismatch or load error -> {e}")
        return None
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    acc = accuracy_score(y_true, y_pred)
    return cm, acc


def scan_and_find(dirs: List[str], data_path: str, class_targets: Dict[str, int], seed: int = 42, image_size: int = 512, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    val_loader, classes = build_val_loader(
        data_path, image_size, seed, batch_size)

    # Map desired counts to indices per current dataset class order
    try:
        desired = {classes.index(name): count for name,
                   count in class_targets.items()}
    except ValueError as e:
        print(
            f"Class names mismatch. Dataset classes={classes}; targets={list(class_targets.keys())}")
        return None

    for d in dirs:
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f)
                 for f in os.listdir(d) if f.endswith('.pth')]
        files.sort()
        print(f"Scanning {d} ({len(files)} files)")
        for f in files:
            out = evaluate_checkpoint(f, val_loader, len(classes), device)
            if out is None:
                continue
            cm, acc = out
            # Check per-class correct counts
            ok = True
            for idx, needed in desired.items():
                correct = int(cm[idx, idx])
                total = int(cm[idx, :].sum())
                if needed != correct:
                    ok = False
                    break
            if ok:
                print(f"FOUND match: {f} | Acc={acc:.4f}")
                # Save stats next to checkpoint
                out_dir = os.path.join(os.path.dirname(f), "Matched_Stats")
                base = os.path.splitext(os.path.basename(f))[0]
                out_dir = os.path.join(out_dir, base)
                os.makedirs(out_dir, exist_ok=True)
                plot_confusion_matrix_png(cm, classes, os.path.join(
                    out_dir, "confusion_matrix.png"), normalize=False, title="Confusion Matrix")
                plot_confusion_matrix_png(cm, classes, os.path.join(
                    out_dir, "confusion_matrix_normalized.png"), normalize=True, title="Confusion Matrix (Normalized)")
                print(f"Saved confusion matrices under: {out_dir}")
                return f
    print("No matching checkpoint found.")
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Find a checkpoint whose confusion matrix matches desired per-class correct counts.")
    ap.add_argument(
        "--data", default=r"E:/Santosh_master_thesis/flat_labeled_data_leaves_trunks")
    ap.add_argument("--leaves", type=int, default=90)
    ap.add_argument("--trunks", type=int, default=90)
    ap.add_argument("--others", type=int, default=86)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--dirs", nargs='*', default=[
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs/All_Epoch_Models",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs_other_emphasis/All_Epoch_Models",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs_other_emphasis",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs_two_stages/All_Epoch_Models",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs_two_stages",
    ])
    args = ap.parse_args()

    # Desired counts by class name; will be mapped to indices per dataset.classes
    targets = {"Leaves": args.leaves,
               "Trunks": args.trunks, "Others": args.others}

    found = scan_and_find(args.dirs, args.data, targets,
                          seed=args.seed, image_size=args.img, batch_size=args.bs)
    if found:
        print(found)


if __name__ == "__main__":
    main()
