# Validate Ultralytics YOLO segmentation labels (polygons).
# Usage:
#   python check_yolo-seg_labels.py "E:/.../data_10_classes.yaml"
#   python check_yolo-seg_labels.py "E:/.../data_10_classes.yaml" --fix
import os, argparse, glob, yaml

# Optional default so you can run without passing the arg
DEFAULT_YAML = "E:/Santosh_master_thesis/DATA_YOLO11_classified_Leaves/data_10_classes.yaml"  # e.g. r"E:\Santosh_master_thesis\DATA_YOLO11_classified_Leaves\data_10_classes.yaml"

def load_yaml(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def img_to_label(p):
    # .../images/.../img.jpg -> .../labels/.../img.txt
    parts = os.path.normpath(p).split(os.sep)
    for i, name in enumerate(parts):
        if name.lower() == "images":
            parts[i] = "labels"
            break
    d = os.sep.join(parts[:-1])
    base = os.path.splitext(parts[-1])[0] + ".txt"
    return os.path.join(d, base)

def is_bad_number(x):
    return (x != x) or (x in (float("inf"), float("-inf")))

def check_label_file(lp, nc, fix=False):
    if not os.path.isfile(lp):
        return {"missing": True}
    lines_out, had_error = [], False
    with open(lp, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]
    for ln in lines:
        if not ln:
            had_error = True
            continue
        parts = ln.split()
        try:
            cls = int(float(parts[0]))
        except Exception:
            had_error = True
            continue
        coords = []
        for p in parts[1:]:
            try:
                v = float(p)
            except Exception:
                v = float("nan")
            coords.append(v)
        # checks
        if cls < 0 or cls >= nc:
            had_error = True
            continue
        if len(coords) < 6 or len(coords) % 2 != 0:
            had_error = True
            continue
        if any(is_bad_number(v) or v < 0.0 or v > 1.0 for v in coords):
            if fix:
                coords = [min(1.0, max(0.0, 0.0 if is_bad_number(v) else v)) for v in coords]
            else:
                had_error = True
                continue
        # keep
        lines_out.append(" ".join([str(cls)] + [f"{v:.6f}" for v in coords]))
    result = {"missing": False, "empty": len(lines) == 0, "fixed": False, "has_errors": had_error}
    if fix:
        with open(lp, "w", encoding="utf-8") as g:
            g.write("\n".join(lines_out) + ("\n" if lines_out else ""))
        result["fixed"] = True
    return result

def collect_images(root):
    globs = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.JPG","*.PNG","*.JPEG"]
    out = []
    for g in globs:
        out += glob.glob(os.path.join(root, "**", g), recursive=True)
    return sorted(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path", nargs=("?" if DEFAULT_YAML else None),
                    default=DEFAULT_YAML, help="path to dataset YAML")
    ap.add_argument("--fix", action="store_true", help="rewrite labels with cleaned polygons")
    ap.add_argument("--show-missing", type=int, default=10, help="print first N missing label paths")
    args = ap.parse_args()

    if not args.yaml_path:
        ap.error("yaml_path is required (or set DEFAULT_YAML in the script)")

    data = load_yaml(args.yaml_path)
    nc = int(data.get("nc", 0) or len(data.get("names", [])))
    assert nc > 0, "YAML must define nc or names."
    splits = {k: data[k] for k in ("train","val","test") if k in data and data[k]}

    # Expand image lists
    images = {k: collect_images(v) for k, v in splits.items()}

    # Report missing labels
    total, missing = 0, []
    for split, imgs in images.items():
        for ip in imgs:
            total += 1
            lp = img_to_label(ip)
            if not os.path.isfile(lp):
                missing.append((ip, lp))

    print(f"Total images: {total}")
    print(f"Missing label files: {len(missing)}")
    for i, (ip, lp) in enumerate(missing[:args.show_missing], 1):
        print(f"[{i}] image: {ip}\n    expect label: {lp}")

    if missing:
        print("\nFix the folder mirroring first: .../images/<split>/sub/.../img.jpg -> "
              ".../labels/<split>/sub/.../img.txt")
        return

    # Validate all labels
    total_imgs = 0
    bad_files = 0
    bad_missing = 0
    max_cls = -1
    for split, imgs in images.items():
        for ip in imgs:
            total_imgs += 1
            lp = img_to_label(ip)
            r = check_label_file(lp, nc, fix=args.fix)
            if r.get("missing") or r.get("empty") or r.get("has_errors"):
                bad_files += 1
            if r.get("missing"):
                bad_missing += 1
            # track max class id
            if os.path.isfile(lp):
                with open(lp, "r", encoding="utf-8") as f:
                    for ln in f:
                        if not ln.strip(): continue
                        try:
                            cls = int(float(ln.split()[0]))
                            max_cls = max(max_cls, cls)
                        except Exception:
                            pass

    print(f"nc in YAML: {nc}, max class in labels: {max_cls}")
    print(f"Images scanned: {total_imgs}, labels missing/empty/bad: {bad_files} (missing: {bad_missing})")
    if max_cls >= nc:
        print(f"[ERROR] Found class id {max_cls} >= nc={nc}. Fix YAML names/nc or relabel.")

if __name__ == "__main__":
    main()