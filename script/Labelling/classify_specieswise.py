import os
import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import shutil

# Paths
model_path = "resnet50v2_specieswise.pth"
label_map_path = "label_mapping.txt"
input_root = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
output_root = "classified"

# Load label map
label_map = {}
species_map = {}
with open(label_map_path) as f:
    for line in f:
        line = line.strip()
        if not line or ": " not in line:
            continue  # skip empty or malformed lines
        idx, name = line.split(": ")
        label_map[int(idx)] = name
        species, category = name.split("/")
        if species not in species_map:
            species_map[species] = {}
        species_map[species][category] = int(idx)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet50(weights="IMAGENET1K_V2")
model.fc = torch.nn.Linear(model.fc.in_features, 9)  # Use 9, not len(label_map)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Predict function
def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    pred_class = torch.argmax(output, dim=1).item()
    return pred_class

# Classify and sort
for species in os.listdir(input_root):
    species_path = os.path.join(input_root, species)
    if not os.path.isdir(species_path):
        continue
    for image_name in os.listdir(species_path):
        image_path = os.path.join(species_path, image_name)
        # Only process image files
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        pred_class = predict(image_path)
        pred_label = label_map[pred_class]
        pred_species, pred_category = pred_label.split("/")

    if pred_species != species:
        print(f"⚠️ Species mismatch for {image_name}: predicted {pred_species}, expected {species}")
        continue

    output_dir = os.path.join(output_root, pred_species, pred_category)
    os.makedirs(output_dir, exist_ok=True)
    shutil.copy(image_path, os.path.join(output_dir, image_name))

print("✅ Classification complete.")
