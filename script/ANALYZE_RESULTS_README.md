# YOLO Results Analyzer

This script provides comprehensive analysis of YOLO training results from any results folder.

## Purpose

After training a YOLO model (YOLOv8, YOLOv11, etc.), you get a results folder containing metrics, plots, and model weights. This script helps you understand all the details in that folder by:

1. **Parsing training metrics** from `results.csv`
2. **Displaying configuration** from `args.yaml`
3. **Analyzing confusion matrices**
4. **Showing best model performance**
5. **Listing all available files**
6. **Providing interpretation guide** for metrics

## Installation

Required Python packages:
```bash
pip install pandas numpy pyyaml
```

Optional (for visualization):
```bash
pip install matplotlib seaborn
```

## Usage

### Basic Usage

```bash
python script/analyze_yolo_results.py <path_to_results_folder>
```

### Example

For Windows paths:
```bash
python script/analyze_yolo_results.py "E:/Santosh_master_thesis/segmentation_project_cleaned_labels/y11s_1024_ft_nomosaic"
```

For Linux/Mac paths:
```bash
python script/analyze_yolo_results.py "/home/user/training_runs/exp1"
```

## What It Analyzes

### 1. Training Configuration
- Model architecture (e.g., yolo11s-seg.pt)
- Training parameters (epochs, batch size, image size)
- Optimizer settings (learning rate, momentum, weight decay)
- Data augmentation settings (mosaic, flip, rotation, etc.)
- Device and workers configuration

### 2. Training Metrics Progression
- Epoch-by-epoch metrics from `results.csv`
- Training losses (box, segmentation, classification)
- Validation metrics (precision, recall, mAP)
- Summary statistics across all epochs
- Best performing epoch identification

### 3. Model Performance
- **Segmentation Metrics (Mask):**
  - Precision (M)
  - Recall (M)
  - mAP@50 (M)
  - mAP@50-95 (M)

- **Detection Metrics (Box):**
  - Precision (B)
  - Recall (B)
  - mAP@50 (B)
  - mAP@50-95 (B)

- **Loss Values:**
  - Box loss, Segmentation loss, Classification loss
  - Both training and validation losses

### 4. Available Files
Lists all files in the results folder:
- Plots & Visualizations (PNG, JPG)
- Model Weights (.pt files)
- Configuration files (YAML)
- Metrics & Data (CSV, JSON, TXT)

### 5. Interpretation Guide
Explains what each metric means and how to interpret the results.

## Understanding the Output

### Key Metrics

**mAP (mean Average Precision):**
- `mAP@50`: Average precision at IoU=0.50 (easier threshold)
- `mAP@50-95`: Average precision across IoU 0.50-0.95 (stricter, more comprehensive)
- Higher is better (0-1 or 0-100%)

**Precision:**
- Of all predictions, how many were correct?
- Higher = fewer false positives

**Recall:**
- Of all ground truth objects, how many were detected?
- Higher = fewer missed detections

**Loss:**
- Lower is better
- Indicates how well the model fits the data

### Metrics Suffixes
- `(B)` = Box detection metrics
- `(M)` = Mask segmentation metrics

For segmentation tasks, focus on `(M)` metrics.

## Example Output

The script will display:

```
================================================================================
YOLO RESULTS ANALYZER
================================================================================
Analyzing results from: E:/Santosh_master_thesis/.../y11s_1024_ft_nomosaic
================================================================================

================================================================================
TRAINING CONFIGURATION
================================================================================

📋 Configuration loaded from: args.yaml
  Model                    : yolo11s-seg.pt
  Task                     : segment
  Epochs                   : 40
  Batch Size               : 12
  Image Size               : 1024
  ...

================================================================================
TRAINING METRICS ANALYSIS
================================================================================

📈 Loaded 40 epochs of training data

🏆 Best Epoch: 38 (based on metrics/mAP50-95(M))
   Best mAP50-95: 0.8543

================================================================================
BEST MODEL PERFORMANCE
================================================================================

🏆 Best performance achieved at Epoch: 38

📊 Segmentation Metrics (Mask):
  Precision      : 0.8721 (87.21%)
  Recall         : 0.8456 (84.56%)
  mAP@50         : 0.9012 (90.12%)
  mAP@50-95      : 0.8543 (85.43%)

...
```

## Typical YOLO Results Folder Structure

```
y11s_1024_ft_nomosaic/
├── args.yaml                           # Training configuration
├── results.csv                         # Epoch-by-epoch metrics
├── confusion_matrix.png                # Confusion matrix visualization
├── confusion_matrix_normalized.png     # Normalized confusion matrix
├── F1_curve.png                        # F1 score curve
├── P_curve.png                         # Precision curve
├── R_curve.png                         # Recall curve
├── PR_curve.png                        # Precision-Recall curve
├── results.png                         # Training metrics plots
├── best.pt                             # Best model weights
├── last.pt                             # Last epoch weights
└── weights/
    ├── best.pt                         # Best model (duplicate)
    └── last.pt                         # Last model (duplicate)
```

## Tips

1. **Compare Multiple Runs**: Run this script on different results folders to compare training experiments

2. **Check for Overfitting**: Compare training vs validation losses
   - If validation loss >> training loss → overfitting
   - Consider using more data augmentation or regularization

3. **Best Epoch**: The script identifies the best epoch based on mAP@50-95
   - This is usually the model you should use for inference

4. **Mosaic Augmentation**: Check if `mosaic` is enabled in configuration
   - Often improves performance but can be disabled for fine-tuning

5. **Image Size**: Larger image sizes (e.g., 1024) generally improve accuracy but require more memory

## Troubleshooting

**"Results folder not found"**
- Check the path is correct
- Use quotes around paths with spaces
- Use forward slashes (/) or escaped backslashes (\\\\) on Windows

**"results.csv not found"**
- Make sure the folder contains YOLO training results
- Check if training completed successfully

**Missing packages**
- Install required packages: `pip install pandas numpy pyyaml`

## Integration with Training Scripts

This analyzer works with results from:
- YOLOv8 segmentation training
- YOLOv11 segmentation training
- YOLO detection training
- Any YOLO variant that produces standard results format

It's compatible with training scripts in this repository:
- `script/Labelling/LT_YOLO11_seg_train.py`
- `script/Labelling/yolo11_seg_train_leaves_trunks.py`
- `script/yolo_v8_model_train.py`

## Author Notes

This script was created to help understand YOLO training results for citizen science tree species segmentation projects. It's designed to be:
- **Comprehensive**: Shows all important metrics and files
- **Educational**: Includes interpretation guide
- **Flexible**: Works with different YOLO versions and tasks
- **User-friendly**: Clear output with emojis and formatting

Feel free to modify and extend based on your needs!
