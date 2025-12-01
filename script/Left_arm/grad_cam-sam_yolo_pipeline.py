"""
Grad-CAM++ → Multi-Point Sampling → SAM v1 (ViT-H) → Mask Filtering → YOLO Segmentation Labels
"""

import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm
from shapely.geometry import Polygon
from skimage import measure
from torchvision import transforms
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

# ============= SAM v1 =============
from segment_anything import sam_model_registry, SamPredictor

# ============= GradCAM =============
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import preprocess_image

# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 640
def find_species_folders(root):
    species_dirs = []
    for path, dirs, files in os.walk(root):
        for d in dirs:
            if d[:3].isdigit():   # folder starts with 001, 002, ...
                species_dirs.append(os.path.join(path, d))
    return sorted(species_dirs)

ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
OUT_MASKS = os.path.join(ROOT, "SAMv1_clean_masks")
OUT_OVERLAY = os.path.join(ROOT, "SAMv1_overlays")
OUT_LABELS = os.path.join(ROOT, "YOLO_labels")

os.makedirs(OUT_MASKS, exist_ok=True)
os.makedirs(OUT_OVERLAY, exist_ok=True)
os.makedirs(OUT_LABELS, exist_ok=True)

SAM_CHECKPOINT = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"

NUM_POINTS = 12
TOPK_ACTIVATIONS = 12
MIN_MASK_AREA = 800
MAX_MASK_AREA_RATIO = 0.92
EDGE_TOUCH_TOL = 0.50
MAX_POLY_SIDES = 120

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1. Load Classifier (EffNetV2-S)
# ============================================================

def load_classifier():
    model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 10)
    model.load_state_dict(torch.load("E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Checkpoint_leftarm_4k_oneCLR/best_model_148_0.33.pth", map_location=device))
    model.to(device).eval()
    return model


# ============================================================
# 2. Grad-CAM++
# ============================================================

def generate_gradcam_pp(model, img_tensor, target_class):
    target_layers = [model.features[-1]]  # EffNetV2-S final block

    cam = GradCAMPlusPlus(
    model=model,
    target_layers=target_layers
)

    grayscale_cam = cam(
        input_tensor=img_tensor.unsqueeze(0),
        targets=[ClassifierOutputTarget(target_class)]
    )[0]

    grayscale_cam = cv2.resize(grayscale_cam, (IMG_SIZE, IMG_SIZE))
    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() + 1e-8)

    return grayscale_cam


def sample_cam_points(cam, k=TOPK_ACTIVATIONS):
    ys, xs = np.unravel_index(np.argsort(cam.ravel())[::-1][:k], cam.shape)
    return np.stack([xs, ys], axis=1)


# ============================================================
# 3. SAM v1 Predict
# ============================================================

def run_sam_v1(predictor, image_bgr, points):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)

    pts = points.astype(np.float32)
    lbls = np.ones(len(pts)).astype(np.int32)

    masks, scores, _ = predictor.predict(
        point_coords=pts,
        point_labels=lbls,
        multimask_output=True,
    )

    best_id = np.argmax(scores)
    return masks[best_id]


# ============================================================
# 4. Mask Quality Filtering
# ============================================================

def valid_mask(mask):
    H, W = mask.shape
    area = mask.sum()
    total_area = H * W

    # ---------------------------------------------
    # 1. Reject only extremely small regions
    # ---------------------------------------------
    if area < 800:     # was 5000 → too strict
        return False

    # ---------------------------------------------
    # 2. Reject only extremely large masks
    #    (full image or nearly full)
    # ---------------------------------------------
    if area > 0.92 * total_area:    # was 0.60 → way too strict
        return False

    # ---------------------------------------------
    # 3. Border Touch Score (0 to 1)
    #    compute % of border pixels occupied
    # ---------------------------------------------
    border_pixels = (
        mask[0, :].sum() +
        mask[-1, :].sum() +
        mask[:, 0].sum() +
        mask[:, -1].sum()
    )

    max_border_pixels = (2 * W + 2 * H)
    border_ratio = border_pixels / max_border_pixels

    # Reject only if >50% of the border is covered
    if border_ratio > 0.50:
        return False

    # ---------------------------------------------
    # 4. Optional: remove extremely thin / line-like masks
    # ---------------------------------------------
    if area < 2000 and border_ratio > 0.20:
        # tiny + touching border = mostly noise
        return False

    return True


# ============================================================
# 5. Mask → Polygon → YOLO
# ============================================================

def mask_to_polygon(mask):
    cnts = measure.find_contours(mask.astype(np.uint8), 0.5)
    if len(cnts) == 0:
        return None

    cnt = max(cnts, key=lambda x: len(x))
    poly = Polygon([(int(x), int(y)) for y, x in cnt])

    if not poly.is_valid:
        poly = poly.buffer(0)

    poly = poly.simplify(1.5, preserve_topology=True)
    return poly


def polygon_to_yolo(poly, w, h):
    coords = np.array(poly.exterior.coords)
    return [[x / w, y / h] for x, y in coords]


def save_yolo_label(label_path, class_id, poly_norm):
    with open(label_path, "w") as f:
        f.write(str(class_id))
        for x, y in poly_norm:
            f.write(f" {x:.6f} {y:.6f}")
        f.write("\n")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("Loading classifier...")
    model = load_classifier()

    print("Loading SAM v1 (ViT-H)...")
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
    sam.to(device)
    predictor = SamPredictor(sam)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    summary = {"total": 0, "gradcam_failed": 0, "sam_failed": 0, "saved": 0}

    species_folders = find_species_folders(ROOT)

    for sp_dir in species_folders:
        species_folder = os.path.basename(sp_dir)

        sp_dir = os.path.join(ROOT, species_folder)
        class_id = int(species_folder.split("_")[0]) - 1

        out_mask = os.path.join(OUT_MASKS, species_folder)
        out_overlay = os.path.join(OUT_OVERLAY, species_folder)
        out_label = os.path.join(OUT_LABELS, species_folder)

        os.makedirs(out_mask, exist_ok=True)
        os.makedirs(out_overlay, exist_ok=True)
        os.makedirs(out_label, exist_ok=True)

        for imgname in tqdm(os.listdir(sp_dir), desc=species_folder):
            summary["total"] += 1

            img_path = os.path.join(sp_dir, imgname)
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                continue

            img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            input_tensor = preprocess(img_rgb).to(device)

            with torch.no_grad():
                pred = model(input_tensor.unsqueeze(0))
                target_class = pred.argmax().item()

            cam = generate_gradcam_pp(model, input_tensor, target_class)
            if cam.max() < 0.05:
                summary["gradcam_failed"] += 1
                continue

            points = sample_cam_points(cam, NUM_POINTS)

            try:
                mask = run_sam_v1(predictor, image_bgr, points)
            except:
                summary["sam_failed"] += 1
                continue

            if not valid_mask(mask):
                summary["sam_failed"] += 1
                continue

            # Save mask
            mask_path = os.path.join(out_mask, imgname.replace(".jpg", "_mask.png"))
            cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

            poly = mask_to_polygon(mask)
            if poly is None:
                summary["sam_failed"] += 1
                continue

            h, w = mask.shape
            poly_norm = polygon_to_yolo(poly, w, h)

            lbl_path = os.path.join(out_label, imgname.replace(".jpg", ".txt"))
            save_yolo_label(lbl_path, class_id, poly_norm)

            # overlay
            overlay = image_bgr.copy()
            overlay[mask > 0] = (0.4 * overlay[mask > 0] +
                                 0.6 * np.array([0, 255, 0])).astype(np.uint8)

            cv2.imwrite(os.path.join(out_overlay, imgname), overlay)

            summary["saved"] += 1

    print("\n=========== SUMMARY ===========")
    for k, v in summary.items():
        print(f"{k:20s}: {v}")
    print("================================\n")


if __name__ == "__main__":
    main()
