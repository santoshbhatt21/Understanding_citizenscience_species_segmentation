import os
import re
import matplotlib.pyplot as plt

folder_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point/All_Epoch_Models"  # adjust this

train_loss = {}
val_loss = {}

for filename in os.listdir(folder_path):
    print("Checking:", filename)
    match = match = re.search(r'model_epoch_(\d+)_train_([0-9.]+)_val_([0-9.]+)', filename)

    if match:
        try:
            epoch = int(match.group(1))
            train = float(match.group(2))
            val = float(match.group(3).rstrip('.'))  # <-- Fix
            train_loss[epoch] = train
            val_loss[epoch] = val
        except ValueError as e:
            print(f"Skipping {filename}: {e}")

if train_loss and val_loss:
    epochs = sorted(train_loss.keys())
    train_losses = [train_loss[ep] for ep in epochs]
    val_losses = [val_loss[ep] for ep in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='orange', marker='s')
    plt.title('Training vs Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(folder_path, "train_val_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.show()
else:
    print("❌ No data extracted. Check file naming or regex.")
