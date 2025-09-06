import argparse, random
from pathlib import Path

IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}

def read_labels(lbl_path: Path):
    classes = set()
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): 
            continue
        cid = int(float(line.split()[0]))
        classes.add(cid)
    return classes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, help=".../images")
    ap.add_argument("--labels-root", required=True, help=".../labels (same split layout as images)")
    ap.add_argument("--split", default="train", help="train split name (default: train)")
    ap.add_argument("--target-per-class", type=int, default=0,
                    help="Target images per class (0 = use max class count).")
    ap.add_argument("--out", required=True, help="Path to write balanced train txt, e.g. balanced_train.txt")
    args = ap.parse_args()

    images_root = Path(args.images_root)
    labels_root = Path(args.labels_root)
    split = args.split

    # collect (image -> label file) pairs under split
    imgs = [p for p in (images_root/split).rglob("*") if p.suffix.lower() in IMG_EXTS]
    img2lbl = {}
    for im in imgs:
        # mirror relative path into labels
        rel = im.relative_to(images_root)
        lbl = labels_root / rel.with_suffix(".txt")
        if lbl.exists():
            img2lbl[im] = lbl

    # map class -> list of images that contain it
    per_class = {0:[], 1:[], 2:[]}
    for im, lbl in img2lbl.items():
        present = read_labels(lbl)
        for c in present:
            if c in per_class:
                per_class[c].append(im)

    counts = {c:len(v) for c,v in per_class.items()}
    print("Class image counts:", counts)

    tgt = args.target_per_class or max(counts.values())
    print(f"Target per class: {tgt}")

    # oversample each class to target
    rng = random.Random(1337)
    balanced = []
    for c, lst in per_class.items():
        if not lst:
            continue
        need = tgt
        reps = need // len(lst)
        rem  = need %  len(lst)
        blk  = lst * reps + rng.sample(lst, rem)
        balanced.extend(blk)

    rng.shuffle(balanced)
    # write absolute image paths (Ultralytics supports txt file lists)
    Path(args.out).write_text("\n".join(str(p) for p in balanced) + "\n", encoding="utf-8")
    print(f"Wrote {len(balanced)} lines to {args.out}")

if __name__ == "__main__":
    main()
