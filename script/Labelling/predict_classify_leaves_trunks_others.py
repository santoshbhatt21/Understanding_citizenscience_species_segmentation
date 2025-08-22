import os
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
                        self.samples.append((os.path.join(dirpath, fname), label_id))
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

def main():
    # ==========================
    # Configuration
    # ==========================
    num_classes = 3
    class_names = ["Leaves", "Trunks", "Others"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = "E:/Santosh_master_thesis/checkpoints_efficientnet_leaves_trunks/best_model_20_0.36.pth"
    labeled_data_path = "E:/Santosh_master_thesis/flat_labeled_data_leaves_trunks"
    unlabeled_data_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
    output_csv_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/prediction_metadata_three_classes.csv"

    # ==========================
    # Transforms
    # ==========================
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ==========================
    # Load Model
    # ==========================
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_features, num_classes))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # ==========================
    # Auto Threshold from Validation Set
    # ==========================
    dataset = RecursiveImageFolder(root=labeled_data_path, transform=transform)
    val_indices = list(range(len(dataset)))
    np.random.seed(42)
    np.random.shuffle(val_indices)
    val_size = int(0.3 * len(val_indices))
    val_loader = DataLoader(dataset, batch_size=32, sampler=SubsetRandomSampler(val_indices[:val_size]), num_workers=8)

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
            thresholds[i] = max(mean - std, 0.6)
        else:
            thresholds[i] = 0.6

    print("\n📊 Class-Specific Confidence Thresholds:")
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

                top1 = np.argmax(probs)
                top1_conf = probs[top1]

                if class_names[top1] == "Leaves" and top1_conf >= thresholds[0]:
                    final_class = "Leaves"
                elif class_names[top1] == "Trunks" and top1_conf >= thresholds[1]:
                    final_class = "Trunks"
                else:
                    final_class = "Others"

                results.append([fpath, final_class, round(top1_conf, 4)])

            except Exception as e:
                print(f"Error on {fpath}: {e}")

    # ==========================
    # Save to CSV
    # ==========================
    df = pd.DataFrame(results, columns=["image_path", "predicted_class", "confidence"])
    df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Predictions saved to {output_csv_path}")

if __name__ == '__main__':
    main()
