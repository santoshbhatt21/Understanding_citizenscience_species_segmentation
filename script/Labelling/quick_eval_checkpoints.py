import os
import re
import json
from collections import Counter

import numpy as np
import torch
from torchvision import datasets, transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score


def build_loaders(data_path: str, image_size: int = 512, seed: int = 42, batch_size: int = 32):
    val_tfm = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base = datasets.ImageFolder(data_path, transform=None)
    targets = [s[1] for s in base.samples]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    val_set = datasets.ImageFolder(data_path, transform=val_tfm)
    val_set.samples = [base.samples[i] for i in val_idx]
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    return val_loader, val_set.classes


def load_model(num_classes: int):
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.5),
                                           torch.nn.Linear(in_features, num_classes))
    return model


def eval_ckpt(ckpt_path: str, data_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, classes = build_loaders(data_path)
    model = load_model(len(classes))
    sd = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            logits = model(x)
            y_true.extend(y.numpy().tolist())
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    acc = accuracy_score(y_true, y_pred)
    return cm, acc, classes


def scan_dir(ckpt_dir: str, data_path: str):
    if not os.path.isdir(ckpt_dir):
        print(f"Missing dir: {ckpt_dir}")
        return
    files = [os.path.join(ckpt_dir, f)
             for f in os.listdir(ckpt_dir) if f.endswith('.pth')]
    files.sort()
    for f in files:
        cm, acc, classes = eval_ckpt(f, data_path)
        # Compute recalls as correct/total per class (diagonal / row sum)
        recalls = []
        for i in range(cm.shape[0]):
            total = int(cm[i, :].sum())
            correct = int(cm[i, i])
            recalls.append((correct, total))
        parts = [f"{cls} {c}/{t}" for cls, (c, t) in zip(classes, recalls)]
        print(f"{os.path.basename(f)} | Acc={acc:.4f} | " + " | ".join(parts))


if __name__ == "__main__":
    # Adjust these if needed
    DATA_PATH = r"E:/Santosh_master_thesis/flat_labeled_data_leaves_trunks"
    CANDIDATES = [
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs/All_Epoch_Models",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs_other_emphasis/All_Epoch_Models",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs_other_emphasis",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs_two_stages/All_Epoch_Models",
        r"E:/Santosh_master_thesis/Checkpoints_tree_organs_two_stages",
    ]
    for d in CANDIDATES:
        print(f"\n== Scanning {d} ==")
        scan_dir(d, DATA_PATH)
