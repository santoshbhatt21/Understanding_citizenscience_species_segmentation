import os, glob, csv, math, argparse

def shoelace_area_norm(coords):
    # coords: [x1,y1,x2,y2,...] normalized 0..1
    if len(coords) < 6 or len(coords) % 2: return 0.0
    xs = coords[0::2]; ys = coords[1::2]
    n = len(xs)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += xs[i] * ys[j] - xs[j] * ys[i]
    return abs(area) * 0.5

def border_fraction(coords, tol=1e-3):
    xs = coords[0::2]; ys = coords[1::2]
    on_border = 0
    for x,y in zip(xs, ys):
        if abs(x-0.0) <= tol or abs(x-1.0) <= tol or abs(y-0.0) <= tol or abs(y-1.0) <= tol:
            on_border += 1
    return on_border / max(1, len(xs))

def parse_line(line):
    parts = line.strip().split()
    if not parts: return None, []
    cls = int(float(parts[0]))
    nums = [float(p) for p in parts[1:]]
    return cls, nums

def scan(labels_root, report_path, max_area=0.60, min_area=1e-5, max_border_frac=0.20, tol=1e-3):
    files = sorted(glob.glob(os.path.join(labels_root, "**", "*.txt"), recursive=True))
    rows = []
    bad_files = set()
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            lines = [ln for ln in (l.strip() for l in fh) if ln]
        if not lines:
            rows.append([f, "", "", "", "", "", "EMPTY_FILE"])
            bad_files.add(f); continue
        for i, ln in enumerate(lines):
            try:
                cls, nums = parse_line(ln)
            except Exception:
                rows.append([f, i, "", "", "", "", "PARSE_ERROR"]); bad_files.add(f); continue
            flags = []
            if len(nums) < 6 or len(nums) % 2 != 0:
                flags.append("BAD_COORD_COUNT")
            # basic range check
            if any((c < -1e-3 or c > 1+1e-3 or math.isnan(c) or math.isinf(c)) for c in nums):
                flags.append("OUT_OF_RANGE")
            area = shoelace_area_norm(nums)
            if area > max_area: flags.append("AREA_TOO_BIG")
            if area < min_area: flags.append("AREA_TOO_SMALL")
            bfrac = border_fraction(nums, tol=tol)
            if bfrac > max_border_frac: flags.append("BORDER_HUGGING")
            if flags: bad_files.add(f)
            rows.append([f, i, cls, len(nums)//2, f"{bfrac:.3f}", f"{area:.3f}", "|".join(flags) or "OK"])
    # write csv
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["file","line_idx","cls","num_points","border_frac","area_norm","flags"])
        w.writerows(rows)
    # summary
    total = len(files)
    print(f"Scanned {total} label files.")
    print(f"Flagged {len(bad_files)} files with at least one issue.")
    print(f"Report saved: {report_path}")
    if bad_files:
        print("Examples:")
        for ex in list(bad_files)[:10]:
            print(" -", ex)

if __name__ == "__main__":
    ap = argparse.ArgumentParser("YOLO-seg label sanity checker")
    ap.add_argument("--labels", required=True, help="Root folder of YOLO segmentation labels (*.txt).")
    ap.add_argument("--report", required=True, help="CSV path to save the findings.")
    ap.add_argument("--max-area", type=float, default=0.60, help="Max allowed polygon area in normalized units.")
    ap.add_argument("--min-area", type=float, default=1e-5, help="Min allowed polygon area.")
    ap.add_argument("--max-border-frac", type=float, default=0.20, help="Max fraction of vertices on image border.")
    ap.add_argument("--tol", type=float, default=1e-3, help="Tolerance for considering a vertex on the border.")
    args = ap.parse_args()
    scan(args.labels, args.report, args.max_area, args.min_area, args.max_border_frac, args.tol)
