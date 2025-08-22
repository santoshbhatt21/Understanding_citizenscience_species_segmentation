import os
import torch
from torchvision import models, transforms
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
import shutil

# --- Config ---
model_path = "E:/Santosh_master_thesis/checkpoints_efficientnet_leaves_trunks/best_model_16_0.21.pth"
image_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
output_csv = "predicted_labels.csv"
output_split_dir = "split_output_images_with_leaves_trunks_uncertain"
uncertainty_threshold = 0.6  # Below this, mark as Uncertain
difference_margin = 0.15     # If both classes are too close, mark as Uncertain

# --- Define Transforms (match your training transforms) ---
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

from torchvision import models

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.efficientnet_v2_s(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 2)  # 2 classes: Leaves, Trunks
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --- Class Mapping ---
class_names = {0: "Leaves", 1: "Trunks", 2: "Uncertain"}

# --- Create Output Dir ---
for name in class_names.values():
    os.makedirs(os.path.join(output_split_dir, name), exist_ok=True)

# --- Inference with Uncertainty ---
results = []

with torch.no_grad():
    for filename in tqdm(os.listdir(image_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_dir, filename)
        image = Image.open(img_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0].cpu()

        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()
        prob_diff = abs(probs[0] - probs[1]).item()

        # Uncertainty rule
        if confidence < uncertainty_threshold or prob_diff < difference_margin:
            pred_label = 2  # Uncertain
        else:
            pred_label = pred_class

        # Save result
        results.append({
            "filename": filename,
            "predicted_label": pred_label,
            "predicted_class": class_names[pred_label],
            "confidence": confidence
        })

        # Copy to output folder
        shutil.copy(img_path, os.path.join(output_split_dir, class_names[pred_label], filename))

# --- Save CSV ---
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n✅ Results saved to: {output_csv}")
print(f"✅ Images split into: {output_split_dir}")
