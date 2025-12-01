"""
grad_cam_sam_pipeline.py
Clean Grad-CAM → Point Sampling → SAM Segmentation Pipeline
"""
import glob
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# ============================================================
#               CONFIGURATION (CHANGE YOUR PATHS HERE)
# ============================================================

IMAGE_FOLDER = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
MASK_OUTPUT_FOLDER = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Masks_SAM_generated"
CLASSIFIER_CKPT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/script/Left_arm/Checkpoint_leftarm_4k_oneCLR/best_model_148_0.33.pth"
SAM_CKPT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"

num_classes = 10     # your dataset has 10 species
img_size = 384       # EfficientNetV2-S size used during CAM

# def main():
#     for img_path in all_image_paths:
#         img_bgr = cv2.imread(img_path)
#         final_mask = generate_final_mask(model, sam_predictor, img_bgr)
#         save_mask(final_mask, ...)

# ============================================================
#        STEP 1 — Rebuild EfficientNetV2-S Classifier
# ============================================================

def load_classifier():
    print("Loading EfficientNetV2-S classifier...")

    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)

    # Replace classifier head
    in_feats = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_feats, num_classes)

    # Load state_dict checkpoint
    state = torch.load(CLASSIFIER_CKPT, map_location="cpu")
    model.load_state_dict(state)

    model = model.cuda().eval()
    print("✔ Classifier loaded.")
    return model


# ============================================================
#                      GRAD-CAM GENERATION
# ============================================================

def generate_gradcam(model, img_tensor, target_class=0):
    target_layer = model.features[-1]

    # Newer pytorch-grad-cam API: no use_cuda argument
    cam = GradCAM(model=model, target_layers=[target_layer])

    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=img_tensor.unsqueeze(0), targets=targets)[0]
    return grayscale_cam


# ============================================================
#                HOTSPOT EXTRACTION (largest CAM region)
# ============================================================

def extract_largest_cam_cluster(cam, thr=150):

    cam_norm = (cam * 255).astype(np.uint8)
    _, binary = cv2.threshold(cam_norm, thr, 255, cv2.THRESH_BINARY)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    if n_labels <= 1:
        return None

    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    hotspot = (labels == largest).astype(np.uint8) * 255
    return hotspot


# ============================================================
#                POINT SAMPLING INSIDE HOTSPOT
# ============================================================

def sample_points_from_hotspot(mask, num_points=3):
    kernel = np.ones((7, 7), np.uint8)
    interior = cv2.erode(mask, kernel)

    ys, xs = np.where(interior > 0)
    coords = list(zip(xs, ys))

    if len(coords) == 0:
        return []

    if len(coords) < num_points:
        return coords

    pts = [coords[np.random.randint(len(coords))]]

    for _ in range(num_points - 1):
        dists = [np.linalg.norm(np.array(c) - np.array(pts[0])) for c in coords]
        pts.append(coords[int(np.argmax(dists))])

    return pts


# ============================================================
#                    SAM SEGMENTATION
# ============================================================

def run_sam_on_points(predictor, image_bgr, points):
    predictor.set_image(image_bgr)

    pts = np.array(points)
    labels = np.ones(len(points))

    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=labels,
        multimask_output=True
    )

    best_mask = masks[np.argmax(scores)]
    return best_mask.astype(np.uint8) * 255


# ============================================================
#                 MASK REFINEMENT / CLEANING
# ============================================================

def refine_mask(mask):

    H, W = mask.shape

    # Remove border-touching masks
    if np.any(mask[0, :]) or np.any(mask[-1, :]) or np.any(mask[:, 0]) or np.any(mask[:, -1]):
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0

    # Keep only largest component
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if num <= 1:
        return np.zeros_like(mask)

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_id = 1 + np.argmax(areas)

    cleaned = (labels == largest_id).astype(np.uint8) * 255

    # smooth edges
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

    return cleaned


# ============================================================
#                    MAIN PIPELINE FUNCTION
# ============================================================
def postprocess_mask(mask, orig_h, orig_w):
    """
    Makes mask YOLO-compatible:
      - Converts to binary 0/255
      - Ensures uint8
      - Resizes back to original image resolution
    """
    # Normalize to [0,1]
    mask = (mask > 0.5).astype(np.uint8)

    # Resize to original size
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Convert to 0/255
    mask = mask * 255

    return mask

def generate_final_mask(model, sam_predictor, img_bgr):

    h, w = img_bgr.shape[:2]

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = preprocess(img_rgb).cuda()

    # 1. Grad-CAM
    cam = generate_gradcam(model, img_tensor)
    if cam is None or np.max(cam) < 0.01:
        print("   [ERROR] CAM is empty or too weak")
        return None

    # 2. Hotspot extraction
    hotspot = extract_largest_cam_cluster(cam)
    if hotspot is None:
        print("   [ERROR] No hotspot found in CAM")
        return None

    # 3. Point sampling
    points = sample_points_from_hotspot(hotspot)
    if len(points) == 0:
        print("   [ERROR] No points extracted from hotspot")
        return None

    # 4. SAM inference
    raw_mask = run_sam_on_points(sam_predictor, img_bgr, points)
    if raw_mask.sum() == 0:
        print("   [ERROR] SAM returned empty mask")
        return None

    # 5. Refinement
    refined_mask = refine_mask(raw_mask)

    # 6. Postprocess for YOLO: NEW STEP
    final_mask = postprocess_mask(refined_mask, h, w)

    # 7. Size sanity check
    if final_mask.sum() < 5000:
        print("   [ERROR] Mask too small (<5000 pixels)")
        return None

    return final_mask



# ============================================================
#                        ENTRY POINT
# ============================================================

if __name__ == "__main__":

    os.makedirs(MASK_OUTPUT_FOLDER, exist_ok=True)

    # Load classifier
    model = load_classifier()

    # Load SAM
    print("Loading SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
    sam.cuda()
    sam_predictor = SamPredictor(sam)
    print("✔ SAM loaded.")

    # ------- Process all images (recursive) -------
print("Collecting images recursively...")

image_paths = []
for ext in ("*.jpg", "*.jpeg", "*.jpg"):
    image_paths.extend(
        glob.glob(os.path.join(IMAGE_FOLDER, "**", ext), recursive=True)
    )

print(f"Total images found: {len(image_paths)}")

from collections import defaultdict
by_species = defaultdict(list)
for p in image_paths:
    rel = os.path.relpath(p, IMAGE_FOLDER)
    species = rel.split(os.sep)[0]
    by_species[species].append(p)

for species, paths in by_species.items():
    print(f"\nProcessing species: {species} ({len(paths)} images)")
    for img_path in tqdm(paths, desc=f"{species}", unit="img"):
        fname = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        mask = generate_final_mask(model, sam_predictor, img_bgr)
        if mask is None:
            continue

    # species folder name = immediate subfolder under IMAGE_FOLDER
    rel_path = os.path.relpath(img_path, IMAGE_FOLDER)
    parts = rel_path.split(os.sep)
    species = parts[0]  # e.g. "Abies_alba"

    species_mask_dir = os.path.join(MASK_OUTPUT_FOLDER, species + "_mask")
    os.makedirs(species_mask_dir, exist_ok=True)

    save_path = os.path.join(species_mask_dir, f"mask_{fname}")
    cv2.imwrite(save_path, mask)
    print(f" ✔ Saved: {save_path}")
    print("All masks processed.")