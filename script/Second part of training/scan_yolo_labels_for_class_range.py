import glob
bad=[]
for f in glob.glob(r"E:\Santosh_master_thesis\Classified Labels\Abies alba Leaves\obs_3492713_photo_4062718.txt", recursive=True):
    for line in open(f, "r", encoding="utf-8"):
        if not line.strip(): continue
        cid = int(line.split()[0])
        if not (0 <= cid <= 19):
            bad.append((f, cid))
print("Bad label lines:", len(bad))
# Expect: 0
