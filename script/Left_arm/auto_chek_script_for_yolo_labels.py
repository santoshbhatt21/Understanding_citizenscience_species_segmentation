import os

LABEL_ROOT = r"E:/Santosh_master_thesis/Understanding_citizenscience_species_segmentation/Labels_10class_clean_safe_FIXED"

def check_label(file_path):
    with open(file_path, "r") as f:
        parts = f.read().strip().split()

    # empty file
    if len(parts) < 3:
        return False, "Too few numbers"

    # class id is integer
    try:
        cls = int(parts[0])
    except:
        return False, "Class ID not integer"

    coords = parts[1:]
    if len(coords) % 2 != 0:
        return False, "Odd number of coordinates"

    for c in coords:
        try:
            v = float(c)
        except:
            return False, f"Non-numeric value {c}"

        if not (0 <= v <= 1):
            return False, f"Value out of range (0-1): {v}"

    # minimum polygon size
    if len(coords) < 10:
        return False, "Polygon too small (<5 points)"

    return True, "OK"


bad_files = []

for root, _, files in os.walk(LABEL_ROOT):
    for f in files:
        if f.endswith(".txt"):
            fp = os.path.join(root, f)
            ok, msg = check_label(fp)
            if not ok:
                bad_files.append((fp, msg))

print("\n=== BAD FILES ===")
for fp, msg in bad_files:
    print(fp, "→", msg)

print("\nCheck complete.")
