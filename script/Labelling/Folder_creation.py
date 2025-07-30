import os

# === 1. Define your 9 species names ===
species_names = [
    "001_Acer_pseudoplatanus",
    "002_Betula_pendula",
    "003_Fagus_sylvatica",
    "004_Fraxinus_excelsior",
    "005_Larix_decidua",
    "006_Picea_abies",
    "007_Pinus_sylvestris",
    "008_Pseudotsuga_menziesii",
    "009_Quercus_rubra"
]

# === 2. Define subfolders for each species ===
subfolders = ["Leaves", "Trunks", "Others"]

# === 3. Set your root directory where everything will be created ===
root_dir = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labelling_Manual"

# === 4. Create the directory structure ===
for species in species_names:
    species_path = os.path.join(root_dir, species)
    os.makedirs(species_path, exist_ok=True)

    for sub in subfolders:
        os.makedirs(os.path.join(species_path, sub), exist_ok=True)

print("✅ Folder structure created successfully.")
