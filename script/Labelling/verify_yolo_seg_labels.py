import os
import sys

DATASET_ROOT = r"E:/Santosh_master_thesis/DATA_YOLOv8"
LABELS_DIRS = [
    os.path.join(DATASET_ROOT, 'labels', 'train'),
    os.path.join(DATASET_ROOT, 'labels', 'val'),
]
NUM_CLASSES = 3  # 0..2 for Leaves, Others, Trunks


def check_label_file(path):
    issues = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f.readlines()]
    except Exception as e:
        return [f"read_error: {e}"]

    if len(lines) == 0:
        issues.append('empty_file')
        return issues

    for i, ln in enumerate(lines, 1):
        if not ln:
            issues.append(f'empty_line:{i}')
            continue
        parts = ln.split()
        try:
            cls = int(parts[0])
        except Exception:
            issues.append(f'bad_class_token:line{i}')
            continue
        if not (0 <= cls < NUM_CLASSES):
            issues.append(f'class_out_of_range:{cls}:line{i}')
        coords = parts[1:]
        if len(coords) < 6:
            issues.append(f'too_few_coords:{len(coords)}:line{i}')
            continue
        if len(coords) % 2 != 0:
            issues.append(f'odd_coords_count:{len(coords)}:line{i}')
            continue
        try:
            vals = [float(x) for x in coords]
        except Exception:
            issues.append(f'non_numeric_coord:line{i}')
            continue
        for v in vals:
            if v < 0 or v > 1:
                issues.append(f'coord_out_of_range:{v:.3f}:line{i}')
                break
    return issues


def main():
    total = 0
    empty = 0
    bad = 0
    samples = []
    for lbl_dir in LABELS_DIRS:
        if not os.path.isdir(lbl_dir):
            print(f"[!] Missing labels dir: {lbl_dir}")
            continue
        for root, _, files in os.walk(lbl_dir):
            for f in files:
                if not f.lower().endswith('.txt'):
                    continue
                total += 1
                p = os.path.join(root, f)
                issues = check_label_file(p)
                if issues:
                    if 'empty_file' in issues:
                        empty += 1
                    else:
                        bad += 1
                    if len(samples) < 10:
                        samples.append((p, issues))

    print(f"Labels checked: {total}")
    print(f"Empty files:   {empty}")
    print(f"Malformed:     {bad}")
    if samples:
        print("\nExamples with issues:")
        for p, iss in samples:
            print(f" - {p}")
            for it in iss:
                print(f"    * {it}")

    if total == 0:
        print("[!] No label files found. Ensure preprocessing created labels under DATA_YOLOv8/labels.")


if __name__ == '__main__':
    main()
