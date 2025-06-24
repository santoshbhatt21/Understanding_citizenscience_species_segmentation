import os
import re
import matplotlib.pyplot as plt

folder_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Check_Point/All_Epoch_Models"  # adjust this path

train_acc = {}
val_acc = {}

# Read accuracy from filenames
for filename in os.listdir(folder_path):
    print(filename)
    match = re.search(r'(train|val)_(\d+)_(\d+\.\d+)_(\d+\.\d+)', filename)
    if match:
        acc_type, epoch, t_acc, v_acc = match.groups()
        epoch = int(epoch)
        if acc_type == 'train':
            train_acc[epoch] = float(t_acc)
        elif acc_type == 'val':
            val_acc[epoch] = float(v_acc)

# Sort by epoch
epochs = sorted(set(train_acc.keys()).intersection(val_acc.keys()))
train_accs = [train_acc[e] for e in epochs]
val_accs = [val_acc[e] for e in epochs]

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_accs, label='Training Accuracy', marker='o', color='green')
plt.plot(epochs, val_accs, label='Validation Accuracy', marker='s', color='red')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(folder_path, "train_val_accuracy.png"), dpi=300, bbox_inches='tight')
plt.show()
