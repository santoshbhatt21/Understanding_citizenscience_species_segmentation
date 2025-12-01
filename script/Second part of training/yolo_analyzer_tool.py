import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# =====================================================================
# CLEAN + SAFE YOLOv11 ANALYZER (NO OCR, NO PDF, NO CM)
# =====================================================================

class CleanYOLOAnalyzer:
    def __init__(self, results_path):
        self.results_path = Path(results_path)
        if not self.results_path.exists():
            raise FileNotFoundError(results_path)

        self.csv_path = self.results_path / "results.csv"
        self.yaml_path = self.results_path / "args.yaml"

        # output folder
        self.plot_dir = self.results_path / "analysis_plots"
        self.plot_dir.mkdir(exist_ok=True)

        # data holders
        self.metrics_df = None
        self.args = None

    # ---------------------------------------------------------------------

    def load_results_csv(self):
        """Load results.csv safely"""
        if not self.csv_path.exists():
            print(f"❌ ERROR: results.csv not found at {self.csv_path}")
            return

        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                print("❌ ERROR: results.csv is empty!")
                return
            df.columns = df.columns.str.strip()
            self.metrics_df = df
            print("✓ Loaded results.csv")
        except Exception as e:
            print(f"❌ ERROR reading results.csv: {e}")

    # ---------------------------------------------------------------------

    def load_args_yaml(self):
        """Optional: load args.yaml if present"""
        if self.yaml_path.exists():
            try:
                with open(self.yaml_path, "r") as f:
                    self.args = yaml.safe_load(f)
                print("✓ Loaded args.yaml")
            except Exception:
                print("⚠ args.yaml exists but could not be loaded")
        else:
            print("⚠ args.yaml not found")

    # ---------------------------------------------------------------------

    def compute_overall_metrics(self):
        """Compute overall P, R, F1, mAP from results.csv"""
        if self.metrics_df is None:
            print("❌ Cannot compute metrics — results.csv not loaded.")
            return

        df = self.metrics_df

        try:
            P = df["metrics/precision(M)"].iloc[-1]
            R = df["metrics/recall(M)"].iloc[-1]

            F1 = 2 * P * R / (P + R + 1e-9)
            mAP50 = df["metrics/mAP50(M)"].iloc[-1]
            mAP5095 = df["metrics/mAP50-95(M)"].iloc[-1]

            print("\n====== FINAL YOLO METRICS ======")
            print(f"Precision      : {P:.4f}")
            print(f"Recall         : {R:.4f}")
            print(f"F1 Score       : {F1:.4f}")
            print(f"mAP@50         : {mAP50:.4f}")
            print(f"mAP@50-95      : {mAP5095:.4f}")
            print("=================================\n")

        except KeyError:
            print("❌ Missing metrics columns in results.csv")

    # ---------------------------------------------------------------------

    def plot_curves(self):
        """Plot loss/pr/mAP curves safely."""
        if self.metrics_df is None:
            print("❌ Cannot plot — results.csv not loaded.")
            return

        df = self.metrics_df

        print("Generating plots...")

        # LOSS CURVE
        plt.figure(figsize=(10, 5))
        for col in df.columns:
            if "loss" in col:
                plt.plot(df["epoch"], df[col], label=col)
        plt.title("YOLO Loss Curves")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid()
        plt.savefig(self.plot_dir / "loss_curve.png", dpi=200)
        plt.close()

        # PRECISION – RECALL
        if "metrics/precision(M)" in df.columns:
            plt.figure(figsize=(8, 5))
            plt.plot(df["epoch"], df["metrics/precision(M)"], label="Precision")
            plt.plot(df["epoch"], df["metrics/recall(M)"], label="Recall")
            plt.title("Precision & Recall Curve")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid()
            plt.savefig(self.plot_dir / "precision_recall_curve.png", dpi=200)
            plt.close()

        # F1 CURVE
        if "metrics/precision(M)" in df.columns and "metrics/recall(M)" in df.columns:
            P = df["metrics/precision(M)"]
            R = df["metrics/recall(M)"]
            F1 = 2 * P * R / (P + R + 1e-9)

            plt.figure(figsize=(8, 5))
            plt.plot(df["epoch"], F1, label="F1 Score")
            plt.title("F1 Curve")
            plt.xlabel("Epoch")
            plt.legend()
            plt.grid()
            plt.savefig(self.plot_dir / "f1_curve.png", dpi=200)
            plt.close()

        # mAP CURVE
        plt.figure(figsize=(8, 5))
        for col in ["metrics/mAP50(M)", "metrics/mAP50-95(M)"]:
            if col in df.columns:
                plt.plot(df["epoch"], df[col], label=col)
        plt.title("mAP Curve")
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid()
        plt.savefig(self.plot_dir / "map_curve.png", dpi=200)
        plt.close()

        print("✓ All plots saved to:", self.plot_dir)

    # ---------------------------------------------------------------------

    def run(self):
        self.load_args_yaml()
        self.load_results_csv()
        self.compute_overall_metrics()
        self.plot_curves()
        print("\n✨ Analysis complete. No errors.\n")


# =====================================================================
# MAIN
# =====================================================================

def main():
    folder = input(
        "Enter YOLOv11 results folder path (or press ENTER for default): "
    ).strip()

    if folder == "":
        folder = r"E:\Santosh_master_thesis\segmentation_project_cleaned_labels\y11s_1024_ft_nomosaic"
        print("Using default folder:", folder)

    analyzer = CleanYOLOAnalyzer(folder)
    analyzer.run()



if __name__ == "__main__":
    main()
