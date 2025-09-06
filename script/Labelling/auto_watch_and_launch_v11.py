import csv
import os
import subprocess
import time
from pathlib import Path

# Paths
RUN_DIR = Path("segmentation_project_0") / "yolov8n_seg_896_retina"
RESULTS_CSV = RUN_DIR / "results.csv"
SCRIPT = Path(__file__).parent / "LOT_python yolo_v8_seg_train.py"

# Config for next run
NEXT_MODEL_FAMILY = "11"  # YOLOv11
NEXT_MODEL_SIZE = "n"     # change to 's' for stronger model
ENV = os.environ.copy()
ENV["MODEL_FAMILY"] = NEXT_MODEL_FAMILY
ENV["MODEL_SIZE"] = NEXT_MODEL_SIZE

# Conditions
TARGET_EPOCH = 10
POLL_SEC = 60


def read_last_epoch(csv_path: Path):
    if not csv_path.exists():
        return None
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            last = None
            for row in r:
                last = row
            if last and "epoch" in last:
                try:
                    return int(float(last["epoch"]))
                except Exception:
                    return None
    except Exception:
        return None
    return None


def main():
    print(f"[WATCH] Watching {RESULTS_CSV} for epoch >= {TARGET_EPOCH}")
    launched = False
    while True:
        epoch = read_last_epoch(RESULTS_CSV)
        if epoch is not None:
            print(f"[WATCH] Current epoch: {epoch}")
        else:
            print("[WATCH] No results yet…")

        if epoch is not None and epoch >= TARGET_EPOCH and not launched:
            print("[WATCH] Target reached. Launching YOLOv11 run…")
            # Launch new run with env overrides for v11
            subprocess.Popen([
                "C:/Users/scanman/anaconda3/Scripts/conda.exe", "run", "-p", "C:/Users/scanman/anaconda3",
                "--no-capture-output", "python", str(SCRIPT)
            ], env=ENV, cwd=str(Path(__file__).parent))
            launched = True
            print("[WATCH] Launched.")
            # Continue to watch, but no further launches
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
