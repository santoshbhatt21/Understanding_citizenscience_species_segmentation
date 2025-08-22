import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from PIL import Image, UnidentifiedImageError

# ==========================
# RecursiveImageFolder
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


if __name__ == "__main__":
    # ==========================
    # Configuration
    # ==========================
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "E:/Santosh_master_thesis/checkpoints_efficientnet_leaves_trunks/best_model_19_0.36.pth"
    data_path = "E:/Santosh_master_thesis/flat_labeled_data_leaves_trunks"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ==========================
    # Load model
    # ==========================
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(
        0.6), nn.Linear(in_features, num_classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # ==========================
    # Load dataset
    # ==========================
    dataset = RecursiveImageFolder(root=data_path, transform=transform)
    indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(indices)
    val_size = int(0.3 * len(indices))
    val_idx = indices[:val_size]
    val_loader = DataLoader(dataset, batch_size=32,
                            sampler=SubsetRandomSampler(val_idx), num_workers=4)

    # ==========================
    # Run inference
    # ==========================
    confidences = []
    correct_flags = []

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)

            confidences.extend(conf.cpu().numpy())
            correct_flags.extend((preds == labels).cpu().numpy())

    # ==========================
    # Analyze thresholds
    # ==========================
    df = pd.DataFrame({"confidence": confidences, "correct": correct_flags})
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]

    print("\n📊 Threshold Analysis:")
    for t in thresholds:
        filtered = df[df["confidence"] >= t]
        if len(filtered) == 0:
            continue
        acc = filtered["correct"].mean()
        print(
            f"Threshold: {t:.2f} | Accuracy: {acc:.3f} | Samples: {len(filtered)}")

    # ==========================
    # Save CSV
    # ==========================
    df.to_csv("confidence_analysis.csv", index=False)
    print("✅ Saved confidence report to: confidence_analysis.csv")
    output_csv = "predictions.csv"
