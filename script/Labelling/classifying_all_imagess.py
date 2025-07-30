import os
import torch
from torchvision import models, transforms
from PIL import Image
import shutil

# === CONFIG ===
model_path = "E:/Santosh_master_thesis/leaf_trunk_species_classifier_27.pth"
image_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
output_dir = "classified_images"
img_size = 256

# === CLASS INDEX TO (SPECIES, CATEGORY) MAPPING ===
# NOTE: Update based on your actual class_to_idx
idx_to_class = {
    0: ('001_Acer_pseudoplatanus', 'Leaves'),
    1: ('001_Acer_pseudoplatanus', 'Others'),
    2: ('001_Acer_pseudoplatanus', 'Trunks'),
    3: ('002_Betula_pendula', 'Leaves'),
    4: ('002_Betula_pendula', 'Others'),
    5: ('002_Betula_pendula', 'Trunks'),
    6: ('003_Fagus_sylvatica', 'Leaves'),
    7: ('003_Fagus_sylvatica', 'Others'),
    8: ('003_Fagus_sylvatica', 'Trunks'),
    9: ('004_Fraxinus_excelsior', 'Leaves'),
    10: ('004_Fraxinus_excelsior', 'Others'),
    11: ('004_Fraxinus_excelsior', 'Trunks'),
    12: ('005_Larix_decidua', 'Leaves'),
    13: ('005_Larix_decidua', 'Others'),
    14: ('005_Larix_decidua', 'Trunks'),
    15: ('006_Picea_abies', 'Leaves'),
    16: ('006_Picea_abies', 'Others'),
    17: ('006_Picea_abies', 'Trunks'),
    18: ('007_Pinus_sylvestris', 'Leaves'),
    19: ('007_Pinus_sylvestris', 'Others'),
    20: ('007_Pinus_sylvestrisPinus', 'Trunks'),
    21: ('008_Pseudotsuga_menziesii', 'Leaves'),
    22: ('008_Pseudotsuga_menziesii', 'Others'),
    23: ('008_Pseudotsuga_menziesii', 'Trunks'),
    24: ('009_Quercus_rubra', 'Leaves'),
    25: ('009_Quercus_rubra', 'Others'),
    26: ('009_Quercus_rubra', 'Trunks'),
}

# === CREATE OUTPUT DIRS ===
for species, category in idx_to_class.values():
    os.makedirs(os.path.join(output_dir, species, category), exist_ok=True)

# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, 27)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# === TRANSFORM ===
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# === CLASSIFY AND SORT ===
with torch.no_grad():
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, fname)
                try:
                    img = Image.open(img_path).convert("RGB")
                    img_t = transform(img).unsqueeze(0).to(device)
                    output = model(img_t)
                    pred_idx = torch.argmax(output, dim=1).item()
                    species, category = idx_to_class[pred_idx]
                    dest = os.path.join(output_dir, species, category, fname)
                    shutil.copy(img_path, dest)
                except Exception as e:
                    print(f"❌ Error on {fname}: {e}")

print("✅ All images sorted by species and category.")
