"""
YOLO Results Analyzer
=====================
This script analyzes YOLO training results from a specified folder.
It reads and explains all metrics, plots, and configuration details.

Usage:
    python analyze_yolo_results.py <path_to_results_folder>

Example:
    python analyze_yolo_results.py "E:/Santosh_master_thesis/segmentation_project_cleaned_labels/y11s_1024_ft_nomosaic"

The script will:
1. Parse results.csv (training/validation metrics over epochs)
2. Analyze confusion matrices
3. Extract and display best model performance
4. Show training configuration
5. Generate summary reports and visualizations
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')


class YOLOResultsAnalyzer:
    """Comprehensive analyzer for YOLO training results."""
    
    def __init__(self, results_path: str):
        """
        Initialize the analyzer with path to results folder.
        
        Args:
            results_path: Path to the YOLO training results folder
        """
        self.results_path = Path(results_path)
        if not self.results_path.exists():
            raise FileNotFoundError(f"Results folder not found: {results_path}")
        
        print(f"\n{'='*80}")
        print(f"YOLO RESULTS ANALYZER")
        print(f"{'='*80}")
        print(f"Analyzing results from: {self.results_path}")
        print(f"{'='*80}\n")
        
        # Define key file paths
        self.results_csv = self.results_path / "results.csv"
        self.args_yaml = self.results_path / "args.yaml"
        self.confusion_matrix_png = self.results_path / "confusion_matrix.png"
        self.confusion_matrix_normalized_png = self.results_path / "confusion_matrix_normalized.png"
        
        # Results data
        self.metrics_df = None
        self.args = None
        self.best_epoch = None
        
    def analyze_all(self):
        """Run complete analysis and display all results."""
        
        # 1. Load and display training configuration
        self._analyze_configuration()
        
        # 2. Load and analyze training metrics
        self._analyze_training_metrics()
        
        # 3. Analyze confusion matrix (if available)
        self._analyze_confusion_matrix()
        
        # 4. Display best model performance
        self._display_best_performance()
        
        # 5. List available files and plots
        self._list_available_files()
        
        # 6. Generate summary
        self._generate_summary()
        
    def _analyze_configuration(self):
        """Analyze and display training configuration."""
        print("\n" + "="*80)
        print("TRAINING CONFIGURATION")
        print("="*80)
        
        if self.args_yaml.exists():
            try:
                import yaml
                with open(self.args_yaml, 'r') as f:
                    self.args = yaml.safe_load(f)
                
                print(f"\n📋 Configuration loaded from: {self.args_yaml.name}")
                
                # Display key configuration parameters
                key_params = {
                    'Model': self.args.get('model', 'N/A'),
                    'Task': self.args.get('task', 'N/A'),
                    'Mode': self.args.get('mode', 'N/A'),
                    'Epochs': self.args.get('epochs', 'N/A'),
                    'Batch Size': self.args.get('batch', 'N/A'),
                    'Image Size': self.args.get('imgsz', 'N/A'),
                    'Device': self.args.get('device', 'N/A'),
                    'Workers': self.args.get('workers', 'N/A'),
                    'Optimizer': self.args.get('optimizer', 'N/A'),
                    'Learning Rate (lr0)': self.args.get('lr0', 'N/A'),
                    'Momentum': self.args.get('momentum', 'N/A'),
                    'Weight Decay': self.args.get('weight_decay', 'N/A'),
                }
                
                for param, value in key_params.items():
                    print(f"  {param:25s}: {value}")
                
                # Data configuration
                print(f"\n📊 Dataset Configuration:")
                data_path = self.args.get('data', 'N/A')
                print(f"  Data YAML: {data_path}")
                
                # Augmentation settings
                print(f"\n🔄 Augmentation Settings:")
                aug_params = {
                    'HSV-H': self.args.get('hsv_h', 'N/A'),
                    'HSV-S': self.args.get('hsv_s', 'N/A'),
                    'HSV-V': self.args.get('hsv_v', 'N/A'),
                    'Degrees (rotation)': self.args.get('degrees', 'N/A'),
                    'Translate': self.args.get('translate', 'N/A'),
                    'Scale': self.args.get('scale', 'N/A'),
                    'Flip LR': self.args.get('fliplr', 'N/A'),
                    'Flip UD': self.args.get('flipud', 'N/A'),
                    'Mosaic': self.args.get('mosaic', 'N/A'),
                }
                
                for param, value in aug_params.items():
                    print(f"  {param:25s}: {value}")
                    
            except Exception as e:
                print(f"⚠️  Could not load args.yaml: {e}")
        else:
            print("⚠️  args.yaml not found in results folder")
    
    def _analyze_training_metrics(self):
        """Load and analyze training metrics from results.csv."""
        print("\n" + "="*80)
        print("TRAINING METRICS ANALYSIS")
        print("="*80)
        
        if not self.results_csv.exists():
            print("⚠️  results.csv not found!")
            return
        
        try:
            # Load metrics
            self.metrics_df = pd.read_csv(self.results_csv)
            
            # Clean column names (remove leading/trailing spaces)
            self.metrics_df.columns = self.metrics_df.columns.str.strip()
            
            print(f"\n📈 Loaded {len(self.metrics_df)} epochs of training data")
            print(f"\nAvailable metrics: {', '.join(self.metrics_df.columns.tolist())}")
            
            # Display key metrics progression
            print("\n" + "-"*80)
            print("METRICS PROGRESSION (First 5, Middle 5, Last 5 epochs)")
            print("-"*80)
            
            # Select important columns for display
            display_cols = []
            possible_cols = [
                'epoch', 'train/box_loss', 'train/seg_loss', 'train/cls_loss',
                'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)',
                'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)',
                'val/box_loss', 'val/seg_loss', 'val/cls_loss'
            ]
            
            for col in possible_cols:
                if col in self.metrics_df.columns:
                    display_cols.append(col)
            
            if not display_cols:
                # Fallback: show first few columns
                display_cols = self.metrics_df.columns[:10].tolist()
            
            # Show progression
            total_epochs = len(self.metrics_df)
            if total_epochs <= 15:
                display_df = self.metrics_df[display_cols]
            else:
                # First 5, middle 5, last 5
                first_5 = self.metrics_df.head(5)
                middle_start = total_epochs // 2 - 2
                middle_5 = self.metrics_df.iloc[middle_start:middle_start+5]
                last_5 = self.metrics_df.tail(5)
                display_df = pd.concat([first_5, middle_5, last_5])[display_cols]
            
            print(display_df.to_string(index=False))
            
            # Summary statistics
            print("\n" + "-"*80)
            print("SUMMARY STATISTICS (across all epochs)")
            print("-"*80)
            
            # Find best epoch based on mAP50-95
            map_col = None
            for col in ['metrics/mAP50-95(B)', 'metrics/mAP50-95(M)', 'metrics/mAP50-95']:
                if col in self.metrics_df.columns:
                    map_col = col
                    break
            
            if map_col:
                self.best_epoch = self.metrics_df[map_col].idxmax()
                print(f"\n🏆 Best Epoch: {self.metrics_df.loc[self.best_epoch, 'epoch']:.0f} (based on {map_col})")
                print(f"   Best mAP50-95: {self.metrics_df.loc[self.best_epoch, map_col]:.4f}")
            
            # Display final epoch metrics
            print("\n📊 Final Epoch Metrics:")
            final_epoch = self.metrics_df.iloc[-1]
            
            for col in display_cols:
                if col != 'epoch' and col in final_epoch.index:
                    value = final_epoch[col]
                    if pd.notna(value):
                        print(f"  {col:30s}: {value:.6f}")
            
        except Exception as e:
            print(f"⚠️  Error reading results.csv: {e}")
            import traceback
            traceback.print_exc()
    
    def _analyze_confusion_matrix(self):
        """Analyze confusion matrix if available."""
        print("\n" + "="*80)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*80)
        
        cm_files = [
            self.confusion_matrix_png,
            self.confusion_matrix_normalized_png,
        ]
        
        found_cm = False
        for cm_file in cm_files:
            if cm_file.exists():
                print(f"\n✓ Found: {cm_file.name}")
                found_cm = True
        
        if not found_cm:
            print("\n⚠️  No confusion matrix images found")
            print("   Expected files: confusion_matrix.png, confusion_matrix_normalized.png")
    
    def _display_best_performance(self):
        """Display best model performance summary."""
        print("\n" + "="*80)
        print("BEST MODEL PERFORMANCE")
        print("="*80)
        
        if self.metrics_df is None or self.best_epoch is None:
            print("⚠️  No metrics data available")
            return
        
        best_metrics = self.metrics_df.iloc[self.best_epoch]
        
        print(f"\n🏆 Best performance achieved at Epoch: {best_metrics.get('epoch', 'N/A'):.0f}")
        print("\n" + "-"*80)
        
        # Segmentation metrics (if available)
        print("\n📊 Segmentation Metrics (Mask):")
        mask_metrics = {
            'Precision': 'metrics/precision(M)',
            'Recall': 'metrics/recall(M)',
            'mAP@50': 'metrics/mAP50(M)',
            'mAP@50-95': 'metrics/mAP50-95(M)',
        }
        
        for metric_name, col_name in mask_metrics.items():
            if col_name in best_metrics.index:
                value = best_metrics[col_name]
                if pd.notna(value):
                    print(f"  {metric_name:15s}: {value:.4f} ({value*100:.2f}%)")
        
        # Box detection metrics (if available)
        print("\n📦 Detection Metrics (Box):")
        box_metrics = {
            'Precision': 'metrics/precision(B)',
            'Recall': 'metrics/recall(B)',
            'mAP@50': 'metrics/mAP50(B)',
            'mAP@50-95': 'metrics/mAP50-95(B)',
        }
        
        for metric_name, col_name in box_metrics.items():
            if col_name in best_metrics.index:
                value = best_metrics[col_name]
                if pd.notna(value):
                    print(f"  {metric_name:15s}: {value:.4f} ({value*100:.2f}%)")
        
        # Loss values at best epoch
        print("\n📉 Loss Values (at best epoch):")
        loss_metrics = {
            'Box Loss (train)': 'train/box_loss',
            'Seg Loss (train)': 'train/seg_loss',
            'Cls Loss (train)': 'train/cls_loss',
            'DFL Loss (train)': 'train/dfl_loss',
            'Box Loss (val)': 'val/box_loss',
            'Seg Loss (val)': 'val/seg_loss',
            'Cls Loss (val)': 'val/cls_loss',
            'DFL Loss (val)': 'val/dfl_loss',
        }
        
        for metric_name, col_name in loss_metrics.items():
            if col_name in best_metrics.index:
                value = best_metrics[col_name]
                if pd.notna(value):
                    print(f"  {metric_name:20s}: {value:.6f}")
    
    def _list_available_files(self):
        """List all available files in the results folder."""
        print("\n" + "="*80)
        print("AVAILABLE FILES AND PLOTS")
        print("="*80)
        
        print(f"\n📁 Results folder contains:")
        
        # Categorize files
        categories = {
            'Plots & Visualizations': [],
            'Model Weights': [],
            'Configuration': [],
            'Metrics & Data': [],
            'Other': []
        }
        
        for item in sorted(self.results_path.iterdir()):
            if item.is_file():
                name = item.name
                
                # Categorize
                if name.endswith('.png') or name.endswith('.jpg'):
                    categories['Plots & Visualizations'].append(name)
                elif name.endswith('.pt'):
                    categories['Model Weights'].append(name)
                elif name.endswith('.yaml') or name.endswith('.yml'):
                    categories['Configuration'].append(name)
                elif name.endswith('.csv') or name.endswith('.json') or name.endswith('.txt'):
                    categories['Metrics & Data'].append(name)
                else:
                    categories['Other'].append(name)
        
        for category, files in categories.items():
            if files:
                print(f"\n{category}:")
                for f in files:
                    file_path = self.results_path / f
                    size = file_path.stat().st_size
                    size_str = self._format_size(size)
                    print(f"  ✓ {f:40s} ({size_str})")
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    def _generate_summary(self):
        """Generate overall summary of the results."""
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        summary_lines = []
        
        # Model info
        if self.args:
            model = self.args.get('model', 'Unknown')
            task = self.args.get('task', 'Unknown')
            summary_lines.append(f"Model: {model}")
            summary_lines.append(f"Task: {task}")
            summary_lines.append(f"Image Size: {self.args.get('imgsz', 'N/A')}")
        
        # Training info
        if self.metrics_df is not None:
            total_epochs = len(self.metrics_df)
            summary_lines.append(f"Total Epochs Trained: {total_epochs}")
            
            # Best performance
            if self.best_epoch is not None:
                best_metrics = self.metrics_df.iloc[self.best_epoch]
                best_epoch_num = best_metrics.get('epoch', 'N/A')
                summary_lines.append(f"Best Epoch: {best_epoch_num:.0f}")
                
                # Get best mAP
                for map_col in ['metrics/mAP50-95(M)', 'metrics/mAP50-95(B)', 'metrics/mAP50-95']:
                    if map_col in best_metrics.index and pd.notna(best_metrics[map_col]):
                        map_value = best_metrics[map_col]
                        summary_lines.append(f"Best mAP@50-95: {map_value:.4f} ({map_value*100:.2f}%)")
                        break
                
                # Get precision and recall
                for prec_col in ['metrics/precision(M)', 'metrics/precision(B)', 'metrics/precision']:
                    if prec_col in best_metrics.index and pd.notna(best_metrics[prec_col]):
                        prec_value = best_metrics[prec_col]
                        summary_lines.append(f"Precision: {prec_value:.4f} ({prec_value*100:.2f}%)")
                        break
                
                for rec_col in ['metrics/recall(M)', 'metrics/recall(B)', 'metrics/recall']:
                    if rec_col in best_metrics.index and pd.notna(best_metrics[rec_col]):
                        rec_value = best_metrics[rec_col]
                        summary_lines.append(f"Recall: {rec_value:.4f} ({rec_value*100:.2f}%)")
                        break
        
        print("\n📋 Quick Summary:")
        for line in summary_lines:
            print(f"  • {line}")
        
        print("\n" + "="*80)
        print("INTERPRETATION GUIDE")
        print("="*80)
        print("""
Key Metrics Explained:

🎯 mAP (mean Average Precision):
   - mAP@50: Average precision at IoU threshold of 0.50 (easier to achieve)
   - mAP@50-95: Average precision across IoU thresholds 0.50-0.95 (more strict)
   - Higher is better (range 0-1, or 0-100%)
   - Industry standard for object detection/segmentation

📊 Precision:
   - Of all predicted instances, how many were correct?
   - Precision = True Positives / (True Positives + False Positives)
   - Higher is better (fewer false alarms)

📊 Recall:
   - Of all actual instances, how many did we detect?
   - Recall = True Positives / (True Positives + False Negatives)
   - Higher is better (fewer missed detections)

📉 Loss Values:
   - Box Loss: Error in bounding box predictions
   - Seg Loss: Error in segmentation mask predictions  
   - Cls Loss: Error in class predictions
   - Lower is better (indicates better fit to training/validation data)

Metrics with (B) refer to box detection, (M) refer to mask segmentation.
For segmentation tasks, focus primarily on (M) metrics.
        """)
        
        print("="*80)
        print("Analysis complete!")
        print("="*80 + "\n")


def main():
    """Main function to run the analyzer."""
    
    # Check if path is provided
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide path to results folder")
        print("\nUsage:")
        print("  python analyze_yolo_results.py <path_to_results_folder>")
        print("\nExample:")
        print('  python analyze_yolo_results.py "E:/Santosh_master_thesis/segmentation_project_cleaned_labels/y11s_1024_ft_nomosaic"')
        sys.exit(1)
    
    results_path = sys.argv[1]
    
    try:
        # Create analyzer and run analysis
        analyzer = YOLOResultsAnalyzer(results_path)
        analyzer.analyze_all()
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
