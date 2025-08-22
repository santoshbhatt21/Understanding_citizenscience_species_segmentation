import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, balanced_accuracy_score
from typing import Optional


def plot_confusion_matrix_png(cm: np.ndarray, labels, out_path: str, normalize: bool = False, title: Optional[str] = None):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ImageFolder root")
    ap.add_argument("--ckpt", required=True, help="Path to model .pth")
    ap.add_argument("--out", required=True, help="Output dir to save stats")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--img", type=int, default=512)
    ap.add_argument("--bs", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, class_names = build_val_loader(
        args.data, args.img, args.seed, args.bs)

    model = load_model(len(class_names))
    sd = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    balc = balanced_accuracy_score(y_true, y_pred)

    # Save PNGs
    plot_confusion_matrix_png(cm, class_names, os.path.join(args.out, "confusion_matrix.png"), normalize=False,
                              title="Confusion Matrix")
    plot_confusion_matrix_png(cm, class_names, os.path.join(args.out, "confusion_matrix_normalized.png"), normalize=True,
                              title="Confusion Matrix (Normalized)")

    # Save report JSON
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True)
    report_path = os.path.join(args.out, "classification_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump(report, f, indent=2)

    # Print per-class counts like "Class 90/90"
    parts = []
    for i, name in enumerate(class_names):
        correct = int(cm[i, i])
        total = int(cm[i, :].sum())
        parts.append(f"{name} {correct}/{total}")
    print(f"Acc={acc:.4f} F1m={f1m:.4f} BalAcc={balc:.4f} | " +
          " | ".join(parts))


if __name__ == "__main__":
    main()
