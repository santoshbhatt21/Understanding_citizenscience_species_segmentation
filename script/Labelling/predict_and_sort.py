import os
import torch
import shutil
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

# --- Paths ---
model_path = "resnet50v2_leaf_trunk_others.pth"
input_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
output_root = "classified_output_new"

# --- Class names (must match training) ---
class_names = ['Leaves', 'Trunks', 'Others']
CONFIDENCE_THRESHOLD = 0.65  # 👈 You can adjust this value

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load model ---
model = models.resnet50(weights='IMAGENET1K_V2')
model.fc = torch.nn.Sequential(
    torch.nn.Linear(model.fc.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, len(class_names))
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Create output folders ---
os.makedirs(output_root, exist_ok=True)

# --- Predict and sort with confidence ---
for species in os.listdir(input_root):
    species_path = os.path.join(input_root, species)
    if not os.path.isdir(species_path): continue

    print(f"\n🔎 Processing species: {species}")

    # Create class folders (Leaves, Trunks, Others, Uncertain)
    for cls in class_names + ['Uncertain']:
        os.makedirs(os.path.join(output_root, species, cls), exist_ok=True)

    for img_name in tqdm(os.listdir(species_path)):
        img_path = os.path.join(species_path, img_name)

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, 1)
                confidence = confidence.item()
                predicted_class = class_names[pred.item()]

            # Decide output folder based on confidence
            if confidence >= CONFIDENCE_THRESHOLD:
                target_class = predicted_class
            else:
                target_class = "Uncertain"

            dest_path = os.path.join(output_root, species, target_class, img_name)
            shutil.copy2(img_path, dest_path)

        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")

print("\n✅ Prediction, confidence check, and sorting complete.")
