import os

# === 1. Define your 10 species names ===
class_names = ["001_Abies_alba",
    "002_Acer_pseudoplatanus",
    "003_Betula_pendula",
    "004_Fagus_sylvatica",
    "005_Fraxinus_excelsior",
    "006_Larix_decidua",
    "007_Picea_abies",
    "008_Pinus_sylvestris",
    "009_Pseudotsuga_menziesii",
    "010_Quercus_rubra"
]

# === 2. Define subfolders for each species ===
subfolders = ["Leaves","Others", "Trunks"]

# === 3. Set your root directory where everything will be created ===
root_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labelling_Manual"

# === 4. Create the directory structure ===
for class_name in class_names:
    class_path = os.path.join(root_dir, class_name)
    os.makedirs(class_path, exist_ok=True)

    for sub in subfolders:
        os.makedirs(os.path.join(class_path, sub), exist_ok=True)

print("✅ Folder structure created successfully.")
