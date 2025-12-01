import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# --------------------------------------------------
# PATHS
# --------------------------------------------------
root = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation"
image_root = os.path.join(root, "Data")
mask_root  = os.path.join(root, "Masks")

# All 10 species folders (images)
species_folders = [
    "001_Abies_alba",
    "002_Acer_pseudoplatanus",
    "003_Betula_pendula",
    "004_Fagus_sylvatica",
    "005_Fraxinus_excelsior",
    "006_Larix_decidua",
    "007_Picea_abies",
    "008_Pinus_sylvestris",
    "009_Pseudotsuga_menziesii",
    "010_Quercus_rubra",
]

N_PER_SPECIES = 1           # one example per species
N_SPECIES_PER_ROW = 1       # 1 species per row  → 10 rows
# --------------------------------------------------


def collect_pairs_for_species(img_dir, mask_dir, max_pairs=1):
    """Return up to max_pairs (img_path, mask_path) pairs for a species."""
    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    pairs = []
    for img_name in image_files:
        base, _ = os.path.splitext(img_name)
        img_path = os.path.join(img_dir, img_name)

        mask_path = None
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            cand = os.path.join(mask_dir, "mask_" + base + ext)
            if os.path.exists(cand):
                mask_path = cand
                break

        if mask_path is not None:
            pairs.append((img_path, mask_path))

        if len(pairs) >= max_pairs:
            break

    return pairs


# --------------------------------------------------
# COLLECT ALL PAIRS
# --------------------------------------------------
all_pairs = []   # list of (species_name, img_path, mask_path)

for species in species_folders:
    img_species_dir  = os.path.join(image_root, species)
    mask_species_dir = os.path.join(mask_root, species + "_mask")

    if not os.path.isdir(img_species_dir):
        print(f"WARNING: missing image folder: {img_species_dir}")
        continue
    if not os.path.isdir(mask_species_dir):
        print(f"WARNING: missing mask folder: {mask_species_dir}")
        continue

    pairs = collect_pairs_for_species(img_species_dir, mask_species_dir,
                                      max_pairs=N_PER_SPECIES)
    for img_path, mask_path in pairs:
        all_pairs.append((species, img_path, mask_path))

# Ensure max 10
all_pairs = all_pairs[:10]
if len(all_pairs) == 0:
    raise RuntimeError("No image–mask pairs found.")
if len(all_pairs) < 10:
    print(f"Only found {len(all_pairs)} species with at least one pair.")

# --------------------------------------------------
# PLOT
# --------------------------------------------------
n_rows = 5
n_cols = N_SPECIES_PER_ROW * 2  # 4 columns: (Photo, Mask) × 2 species

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(4 * n_cols, 3 * n_rows),
)

axes = axes.reshape(n_rows, n_cols)

# Colormap and legend handles
cmap = ListedColormap(["#9ecae1", "#1f77b4"])  # [background, target]
legend_handles = [
    Patch(facecolor="#1f77b4", edgecolor="none", label="Target species"),
    Patch(facecolor="#9ecae1", edgecolor="none", label="Background / other species"),
]

N_SPECIES_PER_ROW = 2
n_rows = 5
n_cols = 4  # (photo, mask) * 2 species

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
axes = axes.reshape(n_rows, n_cols)

for idx, (species, img_path, mask_path) in enumerate(all_pairs):
    row = idx // N_SPECIES_PER_ROW      # 0..4
    col_pair = idx % N_SPECIES_PER_ROW  # 0 or 1
    ax_img  = axes[row, 2 * col_pair]
    ax_mask = axes[row, 2 * col_pair + 1]

    img = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if img.size != mask.size:
        mask = mask.resize(img.size, Image.NEAREST)

    mask_arr = np.array(mask)
    mask_bin = (mask_arr > 0).astype(np.uint8)

    # Photograph
    ax_img.imshow(img)
    ax_img.axis("off")
    # Species label on left subplot
    pretty_name = species.split("_", 1)[1].replace("_", " ")
    ax_img.set_ylabel(pretty_name, fontsize=10,
                      rotation=0, labelpad=40, va="center")

    # Mask
    ax_mask.imshow(mask_bin, cmap=cmap, vmin=0, vmax=1)
    ax_mask.axis("off")

# Column titles (Photograph / Mask)
for c_pair in range(N_SPECIES_PER_ROW):
    axes[0, 2 * c_pair].set_title("Photograph", fontsize=12)
    axes[0, 2 * c_pair + 1].set_title("Mask", fontsize=12)

# Overall figure title
fig.suptitle(
    "Examples of citizen science plant photographs and their segmentation masks",
    fontsize=14, y=0.98
)

# Legend at bottom center, outside the grid
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, 0.02),
    frameon=False,
    fontsize=10
)

plt.tight_layout(rect=[0.02, 0.06, 0.98, 0.95])  # leave space for title & legend
plt.show()