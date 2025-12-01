import torch
import pandas as pd
from pathlib import Path
import pprint

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
TRAINING_STATE_PATH = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/script/Left_arm/Checkpoint_leftarm_4k_oneCLR/training_state.pth"  # <-- your path

# -----------------------------------------------------------
# LOAD TRAINING STATE WITH FIX FOR PYTORCH 2.6
# -----------------------------------------------------------
def load_training_state(path):
    print(f"Loading training state: {path}")

    # PyTorch 2.6 fix
    state = torch.load(path, map_location="cpu", weights_only=False)

    print("\n=== Keys in training_state.pth ===")
    for k in state.keys():
        print("  •", k)
    return state

# -----------------------------------------------------------
# EXPORT METRIC LISTS TO CSV
# -----------------------------------------------------------
def export_history_to_csv(state):
    print("\nExtracting per-epoch histories...")

    # Extract all epoch-based lists
    history = {k: v for k, v in state.items() if isinstance(v, list)}

    if not history:
        print("⚠ No list-based histories found.")
        return

    # Pad uneven lists
    max_len = max(len(v) for v in history.values())
    for k in history.keys():
        if len(history[k]) < max_len:
            history[k] = history[k] + [None] * (max_len - len(history[k]))

    df = pd.DataFrame(history)
    out_path = "training_metrics_history.csv"
    df.to_csv(out_path, index=False)

    print(f"✓ Exported full training history → {out_path}")
    print(df.head())

# -----------------------------------------------------------
# SUMMARY FOR THESIS
# -----------------------------------------------------------
def write_summary(state):
    summary_path = "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=== TRAINING SUMMARY ===\n\n")

        if "epoch" in state:
            f.write(f"Final epoch: {state['epoch']}\n")

        if "best_epoch" in state:
            f.write(f"Best epoch: {state['best_epoch']}\n")

        if "best_loss" in state:
            f.write(f"Best validation loss: {state['best_loss']:.4f}\n")

        if "best_metrics" in state:
            bm = state['best_metrics']
            f.write("\n--- Best Epoch Macro Metrics ---\n")
            f.write(f"Macro Precision: {bm.get('macro_precision', 'N/A'):.4f}\n")
            f.write(f"Macro Recall:    {bm.get('macro_recall', 'N/A'):.4f}\n")
            f.write(f"Macro F1:        {bm.get('macro_f1', 'N/A'):.4f}\n")

    print(f"✓ Thesis summary exported → {summary_path}")

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
def main():
    state = load_training_state(TRAINING_STATE_PATH)
    export_history_to_csv(state)
    write_summary(state)

    print("\n=== RAW STATE PREVIEW ===")
    pprint.pprint(state)

if __name__ == "__main__":
    main()
