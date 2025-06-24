import os
import re
import matplotlib.pyplot as plt

# 1. Set the folder path to your saved models
folder_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point/All_Epoch_Models"

# 2. Prepare dictionaries to store losses
train_loss = {}
val_loss = {}

# 3. Loop through files and extract epoch, train loss, val loss
for filename in os.listdir(folder_path):
    match = re.search(r'model_epoch_(\d+)_train_(\d+\.\d+)_val_(\d+\.\d+)\.pth', filename)
    if match:
        epoch, t_loss, v_loss = match.groups()
        epoch = int(epoch)
        train_loss[epoch] = float(t_loss)
        val_loss[epoch] = float(v_loss)

# 4. Sort epochs and prepare lists for plotting
epochs = sorted(train_loss.keys())
train_losses = [train_loss[e] for e in epochs]
val_losses = [val_loss[e] for e in epochs]

# 5. Plot
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label='Training Loss', marker='o', color='green')
plt.plot(epochs, val_losses, label='Validation Loss', marker='s', color='red')
plt.title("Training vs Validation Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(folder_path, "train_val_loss_plot.png"), dpi=300)
plt.show()