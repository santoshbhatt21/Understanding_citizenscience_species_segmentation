# Copilot Project Instructions

Concise, project-specific guidance for AI coding agents working on tree species instance segmentation & classification in citizen‑science imagery.

## High‑Level Architecture & Data Flow
1. Raw images (organized per class) live under `Data/` and other dated folders; masks (SAM/CAM or manual) appear in various `Mask_*` / `Data_Mask*` directories.
2. Preprocessing / label generation scripts in `script/Baseline_pipeline/` convert masks to YOLO segmentation labels (`*.txt` with class_id + normalized polygon coords) and manage class mappings (`classes.json`).
3. Segmentation training uses Ultralytics YOLO (`yolo11seg_train_10_species.py`, `yolo_v8_model_train.py`) with data YAML files (e.g. external path `.../yolo11s_seg_10classes.yaml`). Result artifacts land under `segmentation_project*/` or `runs/`.
4. Classification training (EfficientNetV2-S) in `efficientnet_train_10_species.py` (and variants) applies two-stage fine-tuning (head warm‑up then partial backbone unfreeze), tracks dual metrics (val loss vs macro F1), optional SWA finishing, and persists checkpoints in `Checkpoints_*` directories.
5. Utility scripts in `script/util/` generate plots (loss, accuracy, F1, confusion matrices) and maintenance tasks (CUDA cache, cleaning labels).
6. Visualization helpers (e.g. `overlay_yolo_polygons.py`) overlay YOLO polygons onto images for QA.

## Key Conventions & Patterns
- YOLO segmentation label format: each line `class_id x1 y1 x2 y2 ... xN yN` with all coordinates normalized to `[0,1]`; polygons require ≥3 points.
- Class mapping persistence: `classes.json` written in the labels output directory (see `sam_cam_masks_into_yolo.py`); adding new classes increments IDs deterministically by discovery order.
- Trunk mask conversion (`sam_cam_masks_into_yolo.py`): cleans binary masks (morph open/close), filters by relative area (`min_area_frac`, `max_area_frac`), selects instances by area strategy (`largest|keep_all|hull`), applies contour simplification (`epsilon_frac`), and writes normalized polygons.
- EfficientNet training best model semantics:
  * Root alias `best_model.pth` = lowest validation loss (early stopping patience on val loss).
  * Per-epoch & F1-best snapshots saved in `All_Epoch_Models/` with metric suffixes.
  * SWA may replace root best if it further reduces val loss.
- Mixup + label smoothing interplay: if mixup enabled, label smoothing capped (≤0.01); train metrics log both hard accuracy and mixup-aware accuracy (`train_acc_mix`).
- Evaluation helper (`evaluate`) returns: `val_loss, acc, f1_macro, bal_acc, y_true, y_pred, cm` — reuse this signature for new model types to stay consistent.
- Visualization & metrics: confusion matrices and classification reports stored under `Training_Stats/Best_Epoch/` via helper functions; follow this directory pattern for new model families.
- GPU expectation: scripts assert `torch.cuda.is_available()`; prefer non-blocking tensor transfers and AMP (`GradScaler`) for performance.

## Typical Workflows
- Generate segmentation labels (trunks example):
  ```powershell
  python script/Baseline_pipeline/sam_cam_masks_into_yolo.py --images <img_dir> --masks <mask_dir> --labels <out_labels_dir> --merge-mode largest --min-area-frac 0.001 --epsilon-frac 0.0012 --labels-per-class --save-overlays 25
  ```
- Train YOLO segmentation:
  ```powershell
  python script/Baseline_pipeline/yolo11seg_train_10_species.py
  ```
- Train EfficientNet (two-stage + SWA): simply run `efficientnet_train_10_species.py`; tune paths `DATA_PATH`, `CHECKPOINT_DIR` at top.
- Visual QA of polygons:
  ```powershell
  python script/Baseline_pipeline/overlay_yolo_polygons.py --image <path> --labels <label_txt> --names <names.txt> --save <out.jpg>
  ```

## When Extending
- Add new preprocessing scripts under `script/Baseline_pipeline/` with clear docstring describing label assumptions; reuse helpers (e.g. contour simplification approach) for consistency.
- Preserve best-model naming contract (`best_model.pth` by val loss) so downstream evaluation scripts continue to work.
- For new model types, mirror training stats folder layout: `Checkpoints_<model>/Training_Stats/{tensorboard,Best_Epoch}` and include a `summary.json` capturing key selection criteria.
- Ensure polygon coordinate normalization and minimum point count; reject malformed lines gracefully (see `parse_yolo_seg_txt` pattern in `overlay_yolo_polygons.py`).
- Keep argument names consistent (`--images`, `--masks`, `--labels`) to simplify automated orchestration.

## External Dependencies
- Core libraries: PyTorch / torchvision, ultralytics YOLO, OpenCV (cv2), numpy, scikit-learn, matplotlib, tqdm, PIL, yaml (optional).
- If adding dependencies, document them at the top of the script and ensure they integrate with existing CUDA usage patterns.

## Quality & Metrics Nuances
- Early stopping strictly on validation loss; do not change to F1 without updating summary logic & artifact regeneration.
- Balanced accuracy (`balanced_accuracy_score`) logged; include it for imbalanced multi-class additions.
- Maintain smoothing of curves via `smooth_curve` before saving plot PNGs for consistent visual output.

## Gotchas
- Windows worker issues: reduce `workers` to `0` in YOLO training if dataloader hangs.
- Mask polarity handling: trunk masks often need inversion (`--polarity fg_black` flips black-foreground to usable binary); keep this flag in new mask converters.
- Duplicate image basenames: first occurrence retained; log warnings — follow same indexing pattern to avoid silent overwrites.

---
Feedback welcome: list unclear sections or missing workflows (e.g., multi-class SAM integration), and we will iterate.
