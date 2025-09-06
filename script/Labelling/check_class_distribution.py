import glob, os

labels_root = r"E:\Santosh_master_thesis\DATA_YOLO11_strict_CAM_setting\labels"
bad, counts = [], {i:0 for i in range(20)}

for f in glob.glob(os.path.join(labels_root, "**", "*.txt"), recursive=True):
    with open(f) as fh:
        for line in fh:
            s = line.strip().split()
            if not s:
                continue
            cid = int(s[0])
            if not (0 <= cid < 20):
                bad.append((f, cid))
            counts[cid] = counts.get(cid, 0) + 1

print("Bad labels (out of range IDs):", bad[:5], "… total:", len(bad))
print("Per-class counts:", counts)
