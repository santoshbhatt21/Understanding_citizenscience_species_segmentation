import os
from PIL import Image, UnidentifiedImageError

def is_image_truncated(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Check if image is broken/truncated
        return False
    except (OSError, UnidentifiedImageError):
        return True

def delete_truncated_images(root_folder):
    truncated_images = []
    
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            filepath = os.path.join(dirpath, fname)
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")):
                if is_image_truncated(filepath):
                    truncated_images.append(filepath)

    print(f"Found {len(truncated_images)} truncated images.")

    for path in truncated_images:
        try:
            os.remove(path)
            print(f"Deleted: {path}")
        except Exception as e:
            print(f"Failed to delete {path}: {e}")

    print("Finished deleting truncated images.")

# === USAGE ===
folder_path = "E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Data"
delete_truncated_images(folder_path)
