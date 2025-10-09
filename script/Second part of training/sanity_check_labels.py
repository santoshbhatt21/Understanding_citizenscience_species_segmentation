from pathlib import Path
import argparse

# Simple YOLO segmentation labels sanity checker.
# Use --data-root pointing to a dataset that has images/train, labels/train, images/val, labels/val


def parse_args():
    ap = argparse.ArgumentParser(
        description="Sanity check YOLO seg labels under images/ and labels/.")
    ap.add_argument("--data-root", required=False,
                    default=r"E:/Santosh_master_thesis/DATA_YOLO11_classified_Leaves_Trunks",
                    help="Dataset root containing images/ and labels/ subfolders.")
    ap.add_argument("--split", default="val", choices=["train", "val"],
                    help="Split to scan.")
    return ap.parse_args()


def read_lbl(p):
    out = []
    try:
        for ln in Path(p).read_text().splitlines():
            if not ln.strip():
                continue
            parts = ln.split()
            cid = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            out.append((cid, coords))
    except Exception:
        pass
    return out


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    im_dir = data_root / f"images/{args.split}"
    lb_dir = data_root / f"labels/{args.split}"
    if not im_dir.exists() or not lb_dir.exists():
        print("ERROR: images/ or labels/ split not found.")
        print("Checked:", im_dir)
        print("and:", lb_dir)
        print("Hint: point --data-root to your dataset root, not the training run folder.")
        return

    tot_imgs = tot_inst = bad_norm = max_cid = 0
    examples_too_many = []
    for img in im_dir.rglob("*.*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        rel = img.relative_to(im_dir)
        lbl = lb_dir / rel.with_suffix(".txt")
        if not lbl.exists():
            continue
        L = read_lbl(lbl)
        inst = len(L)
        tot_imgs += 1
        tot_inst += inst
        if inst > 50:
            examples_too_many.append((inst, str(lbl)))
        for cid, coords in L:
            max_cid = max(max_cid, cid)
            # coords must be normalized [0,1]; even length; x1 y1 x2 y2 ...
            if any((c < 0 or c > 1) for c in coords):
                bad_norm += 1
                break

    print(f"Images scanned: {tot_imgs}")
    print(
        f"Total instances: {tot_inst}  (avg {tot_inst/max(1,tot_imgs):.1f} per image)")
    print(f"Max class id: {max_cid}")
    print(f"Files with NON-normalized coords: {bad_norm}")
    print(f"Examples with >50 instances (showing up to 10):")
    for k, (n, p) in enumerate(sorted(examples_too_many, reverse=True)[:10]):
        print(f"  #{k+1}: {n}  -> {p}")


if __name__ == "__main__":
    main()
