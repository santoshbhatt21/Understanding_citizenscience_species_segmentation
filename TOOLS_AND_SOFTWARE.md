# Tools and Software Used in This Thesis

This document provides a comprehensive list of all tools, software, packages, and frameworks used in the "Understanding Citizen Science Species Segmentation" master's thesis project.

## Programming Language

**Python** - Primary programming language for all scripts and models

## Deep Learning Frameworks

### PyTorch Ecosystem
- **PyTorch** (`torch`) - Core deep learning framework
- **TorchVision** (`torchvision`) - Computer vision library with pretrained models and transforms
  - Models: EfficientNet V2 (S and L variants), ResNet50
  - Transforms for data augmentation
  - Pre-trained weights support
- **TorchMetrics** - Metrics computation for model evaluation
- **PyTorch Grad-CAM** (`pytorch_grad_cam`) - Gradient-weighted Class Activation Mapping for model interpretability
  - GradCAM
  - HiResCAM
- **TensorBoard** (`torch.utils.tensorboard`) - Visualization and logging of training metrics

## Computer Vision Models and Frameworks

### YOLO (You Only Look Once)
- **Ultralytics YOLO** (`ultralytics`) - Object detection and instance segmentation
  - YOLOv8 (yolov8s-seg.pt, yolov8l-seg.pt)
  - YOLOv11 (yolo11s-seg.pt, yolo11m-seg.pt) - Latest version for segmentation tasks

### Segment Anything Model (SAM)
- **Segment Anything** (`segment_anything`) - Meta's foundation model for image segmentation
  - SAM (ViT-H variant: sam_vit_h_4b8939.pth)
  - SAM2 (sam2_hiera_large.pt) - Latest version with improved performance
  - SamPredictor for inference

### CNN Architectures
- **EfficientNet V2** (Small and Large variants) - Transfer learning for classification
- **ResNet50** - Deep residual networks for feature extraction

## Data Processing and Analysis

### Scientific Computing
- **NumPy** (`numpy`) - Numerical computing and array operations
- **Pandas** (`pandas`) - Data manipulation and analysis
- **SciPy** (`scipy`) - Scientific computing, statistical functions
  - `scipy.stats` for Spearman and Pearson correlation

### Image Processing
- **OpenCV** (`cv2`) - Computer vision and image processing
- **Pillow (PIL)** (`PIL`) - Python Imaging Library for image operations
  - Image, ImageFile, ImageDraw, UnidentifiedImageError

## Machine Learning Tools

### Scikit-learn
- **scikit-learn** (`sklearn`) - Machine learning utilities
  - `sklearn.metrics` - Classification metrics (accuracy, F1-score, confusion matrix, balanced accuracy)
  - `sklearn.model_selection` - StratifiedShuffleSplit for data splitting
  - `sklearn.utils.class_weight` - Computing class weights for imbalanced datasets

## Visualization

- **Matplotlib** (`matplotlib.pyplot`) - Plotting and visualization
  - `matplotlib.colors` for color operations
- **Seaborn** (`seaborn`) - Statistical data visualization

## Data Collection and APIs

- **pyiNaturalist** (`pyinaturalist`) - API client for iNaturalist biodiversity platform
  - `get_observations` - Fetch species observations
  - `get_taxa` - Retrieve taxonomic information
- **Requests** (`requests`) - HTTP library for API calls and image downloads

## Utilities and Helpers

### Progress and Logging
- **tqdm** - Progress bars for loops and iterations
- **logging** - Python standard logging module

### File and System Operations
- **os** - Operating system interface
- **pathlib** (`Path`) - Object-oriented filesystem paths
- **shutil** - High-level file operations
- **glob** - Unix-style pathname pattern expansion
- **re** - Regular expressions
- **json** - JSON encoding and decoding
- **yaml** - YAML file parsing
- **csv** - CSV file reading and writing

### Concurrency and Performance
- **concurrent.futures** - Thread and process pool executors
  - `ThreadPoolExecutor` for parallel image downloads
- **multiprocessing** - Process-based parallelism

### Other Utilities
- **datetime** - Date and time operations
- **time** - Time-related functions
- **argparse** - Command-line argument parsing
- **copy** - Shallow and deep copy operations
- **collections** - Specialized container datatypes
  - `Counter`, `OrderedDict`, `defaultdict`
- **typing** - Type hints and annotations
- **functools** - Higher-order functions and operations on callable objects
- **contextlib** - Context managers utilities
- **subprocess** - Subprocess management
- **traceback** - Exception traceback handling
- **math** - Mathematical functions

## Hardware Requirements

### GPU/CUDA
- **CUDA-enabled GPU** - Required for efficient training and inference
- PyTorch with CUDA support
- Scripts utilize GPU acceleration via:
  - `torch.cuda.is_available()`
  - Device management: `cuda:0`, `cuda:1`
  - Batch processing optimized for GPU memory
- **cuDNN** - CUDA Deep Neural Network library (via `torch.backends.cudnn.benchmark`)

## Data Sources

- **iNaturalist** - Citizen science platform for species observations
  - Tree species image dataset
  - 10+ species including: Abies alba, Acer pseudoplatanus, Betula pendula, Fagus sylvatica, Fraxinus excelsior, Larix decidua, Picea abies, Pinus sylvestris, Pseudotsuga menziesii, Quercus rubra

## Training Infrastructure

### Optimizers
- **AdamW** - Adam optimizer with weight decay
- **SGD** - Stochastic Gradient Descent (via torch.optim)

### Learning Rate Schedulers
- **OneCycleLR** - One cycle learning rate policy
- **ReduceLROnPlateau** - Reduce learning rate when metric plateaus
- **SWALR** - Stochastic Weight Averaging learning rate scheduler

### Training Techniques
- **Mixed Precision Training** - Using `torch.amp.autocast` and `GradScaler`
- **Stochastic Weight Averaging (SWA)** - `torch.optim.swa_utils.AveragedModel`
- **Batch Normalization Updates** - `update_bn` for SWA
- **Early Stopping** - With configurable patience
- **Data Augmentation**:
  - Random resized crop
  - Random horizontal/vertical flip
  - Random rotation
  - Color jitter
  - Mosaic, mixup, copy-paste (YOLO-specific)
  - Perspective transforms

## Configuration and Data Formats

- **YAML** - Configuration files for dataset paths and hyperparameters
- **JSON** - Metrics storage and experiment tracking
- **CSV** - Results logging

## Model Checkpoints and Weights

- Pre-trained weights from TorchVision (EfficientNet_V2_S_Weights, EfficientNet_V2_L_Weights)
- Custom trained model checkpoints (.pth, .pt files)
- YOLO pre-trained weights (.pt files)
- SAM pre-trained checkpoints

## Development Tools

- **Git** - Version control
- **GitHub** - Repository hosting and collaboration

## Operating System Support

- Windows (indicated by file paths like `E:/...` and `KMP_DUPLICATE_LIB_OK` environment variable)
- Linux/Unix (indicated by server paths like `/mnt/gsdata/...`)

## Key Workflows

1. **Data Collection**: iNaturalist API → Image downloads with multi-threading
2. **Data Preparation**: Image preprocessing, augmentation, and dataset creation
3. **Annotation**: SAM/SAM2 for automated mask generation → YOLO polygon labels
4. **Training**: 
   - Classification models (EfficientNet)
   - Segmentation models (YOLO v8/v11)
5. **Evaluation**: Confusion matrices, F1-scores, precision/recall metrics
6. **Visualization**: Prediction overlays, grad-CAM heatmaps, training curves

## Dataset Classes

The project focuses on tree species classification and segmentation with multiple class configurations:
- **20-class setup**: Multiple tree species
- **3-class setup**: Leaves, Trunks, Others
- **2-class setup**: Leaves and Trunks only

## Summary

This thesis leverages state-of-the-art deep learning frameworks (PyTorch, YOLO, SAM) combined with traditional computer vision tools (OpenCV) and robust data processing pipelines to tackle the challenge of instance segmentation in citizen science imagery. The toolchain emphasizes reproducibility, scalability, and performance optimization for GPU-accelerated training.
