import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_V2_S_Weights
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def val_transform_fn(image_size=640):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def plot_confusion_png(cm: np.ndarray, labels, out_path: str, normalize: bool = False, title: str = None):
    plt.figure(figsize=(12, 10))
    matrix = cm.astype(float)
    if normalize:
        with np.errstate(all='ignore'):
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix = np.divide(matrix, row_sums, out=np.zeros_like(
                matrix), where=row_sums != 0)
    im = plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45, ha='right', fontsize=8)
    plt.yticks(ticks, labels, fontsize=8)
    fmt = ".2f" if normalize else ".0f"
    thresh = (matrix.max() if matrix.size else 0) / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            plt.text(j, i, format(val, fmt), ha="center", va="center", fontsize=10,
                     color="white" if val > thresh else "black")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    if title:
        plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def collect_images_under(dir_path: str):
    imgs = []
    for dp, _, fns in os.walk(dir_path):
        for fn in fns:
            if fn.lower().endswith(ALLOWED_EXTS):
                imgs.append(os.path.join(dp, fn))
    return imgs


def build_mapped_dataset(root: str, label_to_idx: dict):
    """Return list of (path, idx) using label names from training mapping.
    Supports layouts like root/Leaves/<species> and root/Trunks/<species>, or flat root/<label>.
    """
    samples = []
    missing = []
    # Ensure deterministic order by idx
    idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    class_names = [idx_to_label[i] for i in sorted(idx_to_label.keys())]
    for lbl in class_names:
        idx = label_to_idx[lbl]
        added = 0
        # Organ-suffixed label (e.g., "001_Abies_alba leaves")
        if lbl.endswith(' leaves'):
            species = lbl[:-7]
            d = os.path.join(root, 'Leaves', species)
            if os.path.isdir(d):
                for p in collect_images_under(d):
                    samples.append((p, idx))
                    added += 1
        elif lbl.endswith(' trunks'):
            species = lbl[:-7]
            d = os.path.join(root, 'Trunks', species)
            if os.path.isdir(d):
                for p in collect_images_under(d):
                    samples.append((p, idx))
                    added += 1
        # Fallback: flat folder root/<label>
        if added == 0:
            d = os.path.join(root, lbl)
            if os.path.isdir(d):
                for p in collect_images_under(d):
                    samples.append((p, idx))
                    added += 1
        if added == 0:
            missing.append(lbl)
    if not samples:
        raise RuntimeError(
            'No images found matching training labels under the provided data-root.')
    return samples, class_names, missing


def main():
    parser = argparse.ArgumentParser(
        description="Compute and save confusion_matrix.json (and images) from a checkpoint and dataset")
    parser.add_argument('--data-root', type=str, required=True,
                        help='Dataset root containing class folders (Leaves/Trunks/...)')
    parser.add_argument('--ckpt-root', type=str, required=True,
                        help='Checkpoint root (where best_model.pth and Training_Stats exist)')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model .pth; defaults to <ckpt-root>/best_model.pth')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    stats_dir = os.path.join(args.ckpt_root, 'Training_Stats')
    os.makedirs(stats_dir, exist_ok=True)

    # Use training class mapping from summary.json
    summary_path = os.path.join(stats_dir, 'summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing summary.json at {summary_path}")
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    label_to_idx = summary.get('class_to_idx')
    if not label_to_idx:
        raise ValueError('summary.json does not contain class_to_idx')

    all_samples, class_names, missing = build_mapped_dataset(
        args.data_root, label_to_idx)
    if missing:
        print(
            f"Warning: No images found for labels: {missing[:5]}{'...' if len(missing)>5 else ''}")

    # Stratified split on mapped targets
    targets = [y for _, y in all_samples]
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=args.seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    # Build Dataset wrapper (top-level class to avoid Windows pickling issues)
    from PIL import Image

    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, samples, transform=None):
            self.samples = samples
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, i):
            p, y = self.samples[i]
            with Image.open(p) as im:
                im = im.convert('RGB')
                if self.transform:
                    im = self.transform(im)
            return im, y

    vt = val_transform_fn(args.image_size)
    val_samples = [all_samples[i] for i in val_idx]
    val_set = EvalDataset(val_samples, transform=vt)
    # Use 0 workers on Windows to avoid pickling issues in ad-hoc scripts
    num_workers = 0 if os.name == 'nt' else 4
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, len(class_names)))

    model_path = args.model_path or os.path.join(
        args.ckpt_root, 'best_model.pth')
    state = torch.load(model_path, map_location='cpu')
    try:
        model.load_state_dict(state)
    except Exception:
        # If saved with a module wrapper etc.
        model.load_state_dict(
            {k.replace('module.', ''): v for k, v in state.items()}, strict=False)
    model.to(device)
    model.eval()

    # Eval
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    # Save JSON
    cm_path = os.path.join(stats_dir, 'confusion_matrix.json')
    with open(cm_path, 'w', encoding='utf-8') as f:
        json.dump({'labels': class_names, 'matrix': cm.tolist()}, f)

    # Save images
    plot_confusion_png(cm, class_names, os.path.join(
        stats_dir, 'confusion_matrix.png'), normalize=False, title='Confusion Matrix')
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm.astype(float), row_sums, out=np.zeros_like(
            cm, dtype=float), where=row_sums != 0)
    plot_confusion_png(cm_norm, class_names, os.path.join(
        stats_dir, 'confusion_matrix_normalized.png'), normalize=True, title='Confusion Matrix (Normalized)')

    print(f"Saved: {cm_path}")


if __name__ == '__main__':
    main()
