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
import torchvision.transforms.functional as TF

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
    image_size = 640
    # Prediction policy: pure softmax argmax (no thresholds or near-threshold heuristics)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    checkpoints_dir = "E:/Santosh_master_thesis/Checkpoints_LOT_two_stages_10_species_640"
    labeled_data_path = "E:/Santosh_master_thesis/flat_labeled_Leaves_Others_Trunks_1500_images"
    unlabeled_data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
    output_csv_path = "E:/Santosh_master_thesis/prediction_metadata_LOT_10_species.csv"

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

    # No threshold calibration: we will use argmax over softmax probabilities directly

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
                # Simple flip-based TTA: original, hflip, vflip, hvflip
                imgs = [image,
                        TF.hflip(image),
                        TF.vflip(image),
                        TF.vflip(TF.hflip(image))]

                with torch.no_grad():
                    logits_sum = None
                    for im in imgs:
                        t = transform(im).unsqueeze(0).to(device)
                        out = model(t)
                        logits_sum = out if logits_sum is None else (
                            logits_sum + out)
                    output = logits_sum / len(imgs)
                    probs = F.softmax(output, dim=1)[0].cpu().numpy()

                # Argmax-only selection policy (pure softmax probabilities)
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
            "selection_policy": "softmax_argmax_only; no thresholds; TTA = [orig,hflip,vflip,hvflip]",
        }
        import json as _json
        with open(meta_path, "w", encoding="utf-8") as f:
            _json.dump(meta, f, indent=2)
        print(f"Meta saved to {meta_path}")
    except Exception as _e:
        print(f"Warning: failed to save meta file: {_e}")


if __name__ == '__main__':
    main()
