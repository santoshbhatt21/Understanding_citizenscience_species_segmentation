import os
import re
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
import csv

# ==========================
# Define Dataset Class
# ==========================


class RecursiveImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.samples = []
        self.transform = transform
        self.label_map = {}
        label_id = 0

        for class_name in sorted(os.listdir(root)):
            class_path = os.path.join(root, class_name)
            if not os.path.isdir(class_path):
                continue
            self.label_map[class_name] = label_id
            for dirpath, _, filenames in os.walk(class_path):
                for fname in filenames:
                    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.samples.append(
                            (os.path.join(dirpath, fname), label_id))
            label_id += 1

        self.classes = list(self.label_map.keys())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        for _ in range(10):
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label
            except (OSError, UnidentifiedImageError):
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError(f"Could not read image at index {idx}: {path}")


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """Find the best_model_* file with the lowest val loss in the filename.
    Falls back to the most recently modified best_model_* if parsing fails.
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint dir not found: {checkpoint_dir}")

    best_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(
        "best_model_") and f.endswith(".pth")]
    if not best_files:
        raise FileNotFoundError(
            f"No best_model_*.pth found in {checkpoint_dir}")

    loss_pat = re.compile(r"best_model_\d+_([0-9]+\.[0-9]+)\.pth$")
    parsed = []
    for f in best_files:
        m = loss_pat.match(f)
        if m:
            try:
                loss_val = float(m.group(1))
                parsed.append((loss_val, f))
            except ValueError:
                pass

    if parsed:
        parsed.sort(key=lambda x: x[0])  # lowest loss first
        chosen = parsed[0][1]
    else:
        # Fallback to newest by mtime
        best_files.sort(key=lambda f: os.path.getmtime(
            os.path.join(checkpoint_dir, f)), reverse=True)
        chosen = best_files[0]

    return os.path.join(checkpoint_dir, chosen)


def main():
    # ==========================
    # Configuration
    # ==========================
    image_size = 512
    # If confidence for Leaves/Trunks is within this delta of its threshold, accept it
    NEAR_THRESHOLD_DELTA = 0.03  # 3 percentage points
    # Slightly more forgiving near-threshold window for Trunks
    TRUNK_NEAR_THRESHOLD_DELTA = 0.05
    # Additional margin rule (kept from previous): accept if reasonably confident and well separated
    MIN_ACCEPT_CONF = 0.5
    MIN_MARGIN_DIFF = 0.1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    checkpoints_dir = "E:/Santosh_master_thesis/Checkpoints_tree_organs_two_stages"
    labeled_data_path = "E:/Santosh_master_thesis/flat_labeled_data_leaves_trunks"
    unlabeled_data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
    output_csv_path = "E:/Santosh_master_thesis/prediction_metadata_LOT.csv"

    # ==========================
    # Transforms
    # ==========================
    # Match validation-time transforms from training for stable inference
    transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.1)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ==========================
    # Load Model
    # ==========================
    # Derive class order from labeled dataset to match training
    labeled_dataset_for_classes = RecursiveImageFolder(
        root=labeled_data_path, transform=transform)
    class_names = labeled_dataset_for_classes.classes
    num_classes = len(class_names)

    model_path = find_best_checkpoint(checkpoints_dir)
    print(f"Using checkpoint: {model_path}")
    print(f"Class order: {class_names}")

    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, num_classes))
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)

    # Compute indices for special-case fallback
    name_to_idx = {n: i for i, n in enumerate(class_names)}
    leaves_idx = name_to_idx.get("Leaves")
    trunks_idx = name_to_idx.get("Trunks")
    others_idx = name_to_idx.get("Others")

    # ==========================
    # Auto Threshold from Validation Set
    # ==========================
    dataset = labeled_dataset_for_classes
    val_indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(val_indices)
    val_size = int(0.3 * len(val_indices))
    val_loader = DataLoader(dataset, batch_size=32,
                            sampler=SubsetRandomSampler(val_indices[:val_size]), num_workers=4)

    confidences = {i: [] for i in range(num_classes)}
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Auto Threshold - Validation Pass"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            for i in range(inputs.size(0)):
                if preds[i] == labels[i]:
                    confidences[preds[i].item()].append(conf[i].item())

    thresholds = {}
    for i in range(num_classes):
        confs = confidences[i]
        if confs:
            mean = np.mean(confs)
            std = np.std(confs)
            if class_names[i] == "Trunks":
                thresholds[i] = max(mean - 1.5 * std, 0.45)  # more forgiving
            else:
                thresholds[i] = max(mean - std, 0.6)  # stricter
        else:
            thresholds[i] = 0.6

    print("\nClass-Specific Confidence Thresholds:")
    for i, name in enumerate(class_names):
        print(f"{name}: {thresholds[i]:.3f}")

    # ==========================
    # Predict Unlabeled Data
    # ==========================
    results = []
    image_extensions = (".jpg", ".jpeg", ".png")

    for species in os.listdir(unlabeled_data_path):
        species_dir = os.path.join(unlabeled_data_path, species)
        if not os.path.isdir(species_dir):
            continue

        for fname in tqdm(os.listdir(species_dir), desc=f"Predicting {species}"):
            if not fname.lower().endswith(image_extensions):
                continue

            fpath = os.path.join(species_dir, fname)
            try:
                image = Image.open(fpath).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(tensor)
                    probs = F.softmax(output, dim=1)[0].cpu().numpy()

                sorted_idx = np.argsort(probs)[::-1]
                top1, top2 = sorted_idx[0], sorted_idx[1]
                top1_conf = probs[top1]
                top2_conf = probs[top2]

                # Per-class confidences
                leaf_conf = probs[leaves_idx] if leaves_idx is not None else -1.0
                trunk_conf = probs[trunks_idx] if trunks_idx is not None else -1.0
                other_conf = probs[others_idx] if others_idx is not None else -1.0

                # Per-class thresholds (defaults align with earlier auto logic)
                leaf_thr = thresholds.get(
                    leaves_idx, 0.6) if leaves_idx is not None else 0.6
                trunk_thr = thresholds.get(
                    trunks_idx, 0.45) if trunks_idx is not None else 0.45
                other_thr = thresholds.get(
                    others_idx, 0.6) if others_idx is not None else 0.6

                # Near-threshold acceptance windows
                leaf_near_ok = leaves_idx is not None and (
                    leaf_conf >= max(0.0, leaf_thr - NEAR_THRESHOLD_DELTA))
                trunk_near_ok = trunks_idx is not None and (
                    trunk_conf >= max(0.0, trunk_thr - TRUNK_NEAR_THRESHOLD_DELTA))

                # Others must be clearly better than L/T to dominate
                best_lt = max(leaf_conf, trunk_conf)
                others_margin_ok = (other_conf - best_lt) >= MIN_MARGIN_DIFF
                other_ok = (others_idx is not None) and ((other_conf >= other_thr) and (
                    other_conf >= MIN_ACCEPT_CONF) and others_margin_ok)

                # Acceptability of L/T
                leaf_ok = (leaves_idx is not None) and (
                    (leaf_conf >= leaf_thr) or leaf_near_ok)
                trunk_ok = (trunks_idx is not None) and (
                    (trunk_conf >= trunk_thr) or trunk_near_ok)

                # Top-2 trunk rescue: if trunk is second best and close, prefer trunk
                trunk_rescue = False
                if trunks_idx is not None and top2 == trunks_idx:
                    close_gap = (top1_conf - top2_conf) <= 0.05
                    trunk_rescue = (trunk_conf >= max(
                        0.0, trunk_thr - TRUNK_NEAR_THRESHOLD_DELTA)) and close_gap

                # Selection priority: (1) L/T if acceptable (or trunk rescue), (2) Others if clearly better, (3) fallback to argmax
                if trunk_rescue or leaf_ok or trunk_ok:
                    if trunk_rescue:
                        final_class = trunks_idx
                    else:
                        # choose higher-confidence among acceptable L/T
                        if trunk_ok and (trunk_conf >= leaf_conf or not leaf_ok):
                            final_class = trunks_idx
                        else:
                            final_class = leaves_idx
                elif other_ok:
                    final_class = others_idx
                else:
                    final_class = int(np.argmax(probs))

                label_name = class_names[final_class]
                final_conf = float(probs[final_class])
                results.append([fpath, label_name, round(final_conf, 4)])

            except Exception as e:
                print(f"Error on {fpath}: {e}")
    # ==========================
    # Save to CSV
    # ==========================
    df = pd.DataFrame(results, columns=[
                      "image_path", "predicted_class", "confidence"])
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Predictions saved to {output_csv_path}")

    # Save meta info for traceability
    try:
        meta_path = output_csv_path[:-4] + "_meta.json" if output_csv_path.lower(
        ).endswith(".csv") else output_csv_path + ".meta.json"
        meta = {
            "used_checkpoint": model_path,
            "class_order": class_names,
            "thresholds": {class_names[i]: float(thresholds[i]) for i in range(num_classes)},
            "near_threshold_delta": NEAR_THRESHOLD_DELTA,
            "trunk_near_threshold_delta": TRUNK_NEAR_THRESHOLD_DELTA,
            "min_accept_conf": MIN_ACCEPT_CONF,
            "min_margin_diff": MIN_MARGIN_DIFF,
            "selection_policy": "prefer_L/T_near_threshold; require_Others_margin; trunk_top2_rescue",
        }
        import json as _json
        with open(meta_path, "w", encoding="utf-8") as f:
            _json.dump(meta, f, indent=2)
        print(f"Meta saved to {meta_path}")
    except Exception as _e:
        print(f"Warning: failed to save meta file: {_e}")


if __name__ == '__main__':
    main()
