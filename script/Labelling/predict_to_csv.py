import os
import torch
import csv
from torchvision import models, transforms
from torchvision.models import EfficientNet_V2_S_Weights
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

# ========== Configuration ==========
model_path = "E:/Santosh_master_thesis/checkpoints_leaves_trunks_others/best_model_1_0.55.pth"  # path to your trained model
unlabeled_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"  # directory with species folders containing images
output_csv = "./prediction_metadata.csv"
class_names = ["Leaves", "Trunks", "Others"]
image_extensions = (".jpg", ".jpeg", ".png")
confidence_threshold = 0.0  # if you want to exclude low-confidence predictions

# ========== Transforms ==========
transform = transforms.Compose([
    transforms.Resize(544),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== Load Model ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
in_features = model.classifier[1].in_features
model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.4), torch.nn.Linear(in_features, len(class_names)))
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# ========== Prediction Loop ==========
results = []

for species in os.listdir(unlabeled_root):
    species_dir = os.path.join(unlabeled_root, species)
    if not os.path.isdir(species_dir): continue

    for fname in tqdm(os.listdir(species_dir), desc=f"Processing {species}"):
        if not fname.lower().endswith(image_extensions):
            continue

        fpath = os.path.join(species_dir, fname)
        try:
            image = Image.open(fpath).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                probs = F.softmax(output, dim=1)
                confidence, pred_idx = torch.max(probs, 1)

            confidence = confidence.item()
            pred_class = class_names[pred_idx.item()]

            if confidence >= confidence_threshold:
                results.append([fpath, pred_class, round(confidence, 4)])

        except Exception as e:
            print(f"⚠️ Failed on {fpath}: {e}")

# ========== Write to CSV ==========
with open(output_csv, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "predicted_class", "confidence"])
    writer.writerows(results)

print(f"✅ Predictions saved to {output_csv}")
