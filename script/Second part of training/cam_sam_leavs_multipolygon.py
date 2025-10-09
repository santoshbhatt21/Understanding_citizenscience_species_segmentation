import os
import random
import logging
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from segment_anything import sam_model_registry, SamPredictor

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("cam_sam_multi")

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR        = "E:/Santosh_master_thesis/Classified_Leaves"
CHECKPOINT_PATH = "E:/Santosh_master_thesis/Checkpoints_Leaves_OneCycle_F1_Temp_bestCM/best_by_loss_ep25_0.624.pth"
SAM_CHECKPOINT  = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/SAM2/sam_vit_h_4b8939.pth"

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.JPG', '.PNG', '.JPEG')

BATCH_SIZE = 8
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ------------------------------------------------------------
# CAM thresholds (dual)
# ------------------------------------------------------------
THRESHOLD_MODE   = "pct"  # "pct" | "fixed"
MID_CAM_PERCENT  = 85
HIGH_CAM_PERCENT = 95

# ------------------------------------------------------------
# Prompt point sampling
# ------------------------------------------------------------
POS_POINTS_PER_CONTOUR = 5
NEG_POINTS             = 12
NEG_RING_DILATE        = 19
NEG_RING_EXTRA         = 32
DILATE_CORE_PX         = 9
FG_MAX_FRAC            = 0.55
ERODE_KERNEL           = 5

# ------------------------------------------------------------
# Mask post-processing
# ------------------------------------------------------------
MIN_COMPONENT_AREA_FRAC = 0.003
HOLE_FILL_MAX_FRAC      = 0.001
CLOSE_KERNEL            = 3   # 0/1 disables

# ------------------------------------------------------------
# Polygonization / output
# ------------------------------------------------------------
ALLOW_MULTIPOLY        = True
MAX_POLYGONS_PER_IMAGE = 5
MIN_POLYGON_POINTS     = 6
POLY_EPSILON_FRAC      = 0.008
SAVE_EMPTY_LABELS      = False
SAVE_MASKS_PNG         = True

# ------------------------------------------------------------
# Seeds
# ------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ------------------------------------------------------------
# Globals (set at runtime)
# ------------------------------------------------------------
model = None
cam = None
sam = None
predictor = None
transform = None
CLASS_NAMES = []
NUM_CLASSES = 0

# ------------------------------------------------------------
# Init
# ------------------------------------------------------------
def initialize():
    global model, cam, sam, predictor, transform
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])

    if model is None:
        model_local = models.efficientnet_v2_s(weights=None)
        in_f = model_local.classifier[1].in_features
        model_local.classifier = nn.Sequential(nn.Dropout(0.6), nn.Linear(in_f, NUM_CLASSES))
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        new_state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in ckpt.items())
        model_local.load_state_dict(new_state, strict=False)
        model_local.to(DEVICE).eval()
        layer = model_local.features[-1]
        _cam = GradCAM(model=model_local, target_layers=[layer])
        logger.info("Classifier + Grad-CAM initialized.")
        model = model_local
        cam = _cam

    if sam is None:
        if not os.path.isfile(SAM_CHECKPOINT):
            raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CHECKPOINT}")
        _sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        _sam.to(DEVICE)
        _predictor = SamPredictor(_sam)
        sam = _sam
        predictor = _predictor
        logger.info("SAM predictor initialized.")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def percentile_bin(cam_u8, pct):
    thr = np.percentile(cam_u8, pct)
    _, b = cv2.threshold(cam_u8, int(thr), 255, cv2.THRESH_BINARY)
    return b

def get_cam_masks(cam_u8):
    if THRESHOLD_MODE == "pct":
        mid = percentile_bin(cam_u8, MID_CAM_PERCENT)
        high = percentile_bin(cam_u8, HIGH_CAM_PERCENT)
    else:
        _, mid = cv2.threshold(cam_u8, int(MID_CAM_PERCENT), 255, cv2.THRESH_BINARY)
        _, high = cv2.threshold(cam_u8, min(255, int(MID_CAM_PERCENT)+25), 255, cv2.THRESH_BINARY)
    return high, mid

def sample_points_within_contour(contour, n):
    x,y,w,h = cv2.boundingRect(contour)
    if w == 0 or h == 0: return []
    mask = np.zeros((h,w), np.uint8)
    shifted = contour - np.array([[x,y]])
    cv2.drawContours(mask, [shifted], -1, 255, cv2.FILLED)
    ys,xs = np.where(mask==255)
    if len(xs)==0: return []
    if len(xs) <= n:
        idx = range(len(xs))
    else:
        idx = random.sample(range(len(xs)), n)
    return [(int(xs[i]+x), int(ys[i]+y)) for i in idx]

def sample_negative_ring_from_masks(high_bin, mid_bin):
    if NEG_POINTS <= 0: return []
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (NEG_RING_DILATE, NEG_RING_DILATE))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (NEG_RING_DILATE+NEG_RING_EXTRA,
                                                       NEG_RING_DILATE+NEG_RING_EXTRA))
    mid_dil  = cv2.dilate(mid_bin, k1)
    high_dil = cv2.dilate(high_bin, k1)
    ring1 = ((mid_dil>0) & (high_dil==0)).astype(np.uint8)
    ring2 = ((cv2.dilate(mid_bin, k2)>0) & (mid_dil==0)).astype(np.uint8)
    ring = np.clip(ring1+ring2,0,1)
    ys,xs = np.where(ring>0)
    if len(xs)==0: return []
    choose = np.random.choice(len(xs), size=min(NEG_POINTS,len(xs)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in choose]

def remove_small_components(mask01, min_frac):
    h,w = mask01.shape
    min_area = max(1, int(min_frac*h*w))
    num,lab,stats,_ = cv2.connectedComponentsWithStats(mask01.astype(np.uint8), 8)
    out = np.zeros_like(mask01, np.uint8)
    for i in range(1,num):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab==i] = 1
    return out

def fill_small_holes(mask01, max_hole_frac):
    h,w = mask01.shape
    inv = (mask01==0).astype(np.uint8)
    num,lab,stats,_ = cv2.connectedComponentsWithStats(inv, 8)
    out = mask01.copy()
    limit = max(1, int(max_hole_frac*h*w))
    for i in range(1,num):
        if stats[i, cv2.CC_STAT_AREA] <= limit:
            out[lab==i] = 1
    return out

def close_mask(mask01, ksize):
    if not ksize or ksize < 2: return mask01
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(mask01.astype(np.uint8), cv2.MORPH_CLOSE, k)

def clean_mask(mask01):
    m = remove_small_components(mask01, MIN_COMPONENT_AREA_FRAC)
    m = fill_small_holes(m, HOLE_FILL_MAX_FRAC)
    m = close_mask(m, CLOSE_KERNEL)
    return m.astype(np.uint8)

def simplify_contour(cnt, eps_frac):
    if len(cnt) < 3: return cnt
    peri = cv2.arcLength(cnt, True)
    eps = eps_frac * peri
    approx = cv2.approxPolyDP(cnt, eps, True)
    return approx if len(approx) >= 3 else cnt

def contour_to_poly(cnt, w, h):
    if cnt is None or len(cnt)<3: return []
    xs = np.clip(cnt[:,0,0].astype(np.float32), 0, w-1)
    ys = np.clip(cnt[:,0,1].astype(np.float32), 0, h-1)
    keep=[0]
    for i in range(1,len(xs)):
        if xs[i]!=xs[i-1] or ys[i]!=ys[i-1]:
            keep.append(i)
    xs=xs[keep]; ys=ys[keep]
    if len(xs)<3: return []
    xs/=float(w); ys/=float(h)
    flat=[]
    for a,b in zip(xs,ys): flat.extend([float(a), float(b)])
    return flat

def write_polygons(label_path, class_id, polygons):
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    if not polygons and not SAVE_EMPTY_LABELS:
        return
    with open(label_path,"w",encoding="utf-8") as f:
        for poly in polygons:
            f.write(" ".join([str(class_id)] + [f"{v:.6f}" for v in poly]) + "\n")

# ------------------------------------------------------------
# Batch processing
# ------------------------------------------------------------
def process_batch(image_paths, class_id, mask_out_dir, label_out_dir):
    global cam, predictor, transform
    pil_list=[]
    tensors=[]
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
        except Exception as e:
            logger.warning(f"Skip {p}: {e}")
            continue
        pil_list.append((p, im))
        tensors.append(transform(im).unsqueeze(0))
    if not tensors: return
    batch = torch.cat(tensors,0).to(DEVICE)

    cams = cam(
        input_tensor=batch,
        targets=[ClassifierOutputTarget(class_id)] * batch.shape[0],
        eigen_smooth=False,
        aug_smooth=False
    )

    for i,(path,pil_img) in enumerate(pil_list):
        W,H = pil_img.size
        cam_map = cams[i]
        cam_resized = cv2.resize(cam_map, (W,H), interpolation=cv2.INTER_LINEAR)
        cam_u8 = (cam_resized*255).astype(np.uint8)
        high_bin, mid_bin = get_cam_masks(cam_u8)

        high_cnts,_ = cv2.findContours(high_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pos_pts=[]
        for c in high_cnts:
            pos_pts.extend(sample_points_within_contour(c, POS_POINTS_PER_CONTOUR))
        neg_pts = sample_negative_ring_from_masks(high_bin, mid_bin) if pos_pts else []
        all_pts = pos_pts + neg_pts
        labels = [1]*len(pos_pts) + [0]*len(neg_pts)

        refined_mask = np.zeros((H,W), np.uint8)
        if all_pts:
            predictor.set_image(np.array(pil_img))
            masks,scores,logits = predictor.predict(
                point_coords=np.array(all_pts, dtype=np.float32),
                point_labels=np.array(labels, dtype=np.int32),
                multimask_output=True
            )
            if masks is not None and len(masks)>0:
                best = int(np.argmax(scores))
                best_logits = logits[best]
                refined,_,_ = predictor.predict(
                    point_coords=np.array(all_pts, dtype=np.float32),
                    point_labels=np.array(labels, dtype=np.int32),
                    mask_input=best_logits[None,:,:],
                    multimask_output=False
                )
                refined_mask = np.squeeze(refined).astype(np.uint8)

        k_limit = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(DILATE_CORE_PX,DILATE_CORE_PX))
        mid_limit = cv2.dilate((mid_bin>0).astype(np.uint8), k_limit)
        constrained = refined_mask & mid_limit
        cleaned = clean_mask(constrained)

        fg_frac = cleaned.mean()
        if fg_frac > FG_MAX_FRAC:
            er_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ERODE_KERNEL,ERODE_KERNEL))
            for _ in range(6):
                if cleaned.mean() <= FG_MAX_FRAC: break
                cleaned = cv2.erode(cleaned, er_k)
            cleaned = clean_mask(cleaned)

        if SAVE_MASKS_PNG:
            os.makedirs(mask_out_dir, exist_ok=True)
            cv2.imwrite(
                os.path.join(mask_out_dir, f"{os.path.splitext(os.path.basename(path))[0]}.png"),
                (cleaned*255).astype(np.uint8)
            )

        cnts,_ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polys=[]
        if cnts:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            chosen = cnts[:MAX_POLYGONS_PER_IMAGE] if ALLOW_MULTIPOLY else [max(cnts, key=cv2.contourArea)]
            for c in chosen:
                if cv2.contourArea(c) <= 1: continue
                c = simplify_contour(c, POLY_EPSILON_FRAC)
                if len(c) < MIN_POLYGON_POINTS: continue
                poly = contour_to_poly(c, W, H)
                if len(poly) >= 6: polys.append(poly)

        os.makedirs(label_out_dir, exist_ok=True)
        lbl_path = os.path.join(label_out_dir, f"{os.path.splitext(os.path.basename(path))[0]}.txt")
        write_polygons(lbl_path, class_id, polys)
        logger.info(f"{os.path.basename(path)} class={class_id} polys={len(polys)} fg={fg_frac:.3f} pos={len(pos_pts)} neg={len(neg_pts)}")

# ------------------------------------------------------------
# Discover classes & images
# ------------------------------------------------------------
def discover_classes():
    return [d for d in sorted(os.listdir(BASE_DIR)) if os.path.isdir(os.path.join(BASE_DIR,d))]

def discover_images(cls_name):
    cls_path = os.path.join(BASE_DIR, cls_name)
    imgs=[]
    for root,_,files in os.walk(cls_path):
        for f in files:
            if f.lower().endswith(IMAGE_EXTS):
                imgs.append(os.path.join(root,f))
    return sorted(imgs)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    global CLASS_NAMES, NUM_CLASSES
    CLASS_NAMES = discover_classes()
    if not CLASS_NAMES:
        raise RuntimeError("No class folders found.")
    NUM_CLASSES = len(CLASS_NAMES)
    logger.info(f"Classes ({NUM_CLASSES}): {CLASS_NAMES}")
    initialize()

    for class_idx, cls_name in enumerate(CLASS_NAMES):
        imgs = discover_images(cls_name)
        if not imgs:
            logger.info(f"Skip empty class {cls_name}")
            continue
        mask_dir  = os.path.join(BASE_DIR, f"{cls_name}_mask")
        label_dir = os.path.join(BASE_DIR, f"{cls_name}_labels_yolo")
        logger.info(f"\nClass {cls_name} -> id {class_idx} images={len(imgs)}")
        for i in range(0, len(imgs), BATCH_SIZE):
            batch = imgs[i:i+BATCH_SIZE]
            process_batch(batch, class_idx, mask_dir, label_dir)

    logger.info("Done.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal: {e}")