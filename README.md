# Tree species instance segmentation in citizen science imagery

This repository contains code for generating masks for citizen science plants photographs.
I am using this repo to understand the process for the same. 
I am working on my master thesis.
Suggestions/guides are always welcome.

## Tools

### YOLO Results Analyzer
A comprehensive script to analyze and explain YOLO training results from any results folder.

**Location:** `script/analyze_yolo_results.py`

**Usage:**
```bash
python script/analyze_yolo_results.py <path_to_results_folder>
```

**Example:**
```bash
python script/analyze_yolo_results.py "E:/Santosh_master_thesis/segmentation_project_cleaned_labels/y11s_1024_ft_nomosaic"
```

**Features:**
- Parses and displays training metrics from results.csv
- Shows model configuration and hyperparameters
- Identifies best performing epoch
- Displays segmentation and detection metrics
- Lists all available files and visualizations
- Provides interpretation guide for metrics

For detailed documentation, see [ANALYZE_RESULTS_README.md](script/ANALYZE_RESULTS_README.md)
