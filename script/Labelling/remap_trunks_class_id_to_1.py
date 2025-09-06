import os
from typing import Tuple


DATA_ROOT = r"E:\Santosh_master_thesis\DATA_YOLO11_LT\labels"


def remap_file(path: str) -> Tuple[int, int]:
    """
    Remap leading class id 2 -> 1 on each line of a YOLO polygon label file.
    Returns (lines_total, lines_changed).
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    changed = 0
    new_lines = []
    for ln in lines:
        s = ln.strip()
        if not s:
            new_lines.append(ln)
            continue
        # Split only on the first space to keep coordinates untouched
        first, sep, rest = s.partition(" ")
        if first == "2":
            changed += 1
            new_lines.append("1" + (" " + rest if sep else ""))
        else:
            new_lines.append(ln)

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            for nl in new_lines:
                f.write(nl + "\n")

    return (len(lines), changed)


def main():
    total_files = 0
    total_lines = 0
    total_changed = 0

    for split in ("train", "val"):
        trunks_dir = os.path.join(DATA_ROOT, split, "Trunks")
        if not os.path.isdir(trunks_dir):
            continue
        for root, _, files in os.walk(trunks_dir):
            for fn in files:
                if not fn.lower().endswith(".txt"):
                    continue
                fp = os.path.join(root, fn)
                total_files += 1
                lines, changed = remap_file(fp)
                total_lines += lines
                total_changed += changed

    print(f"Processed files: {total_files}")
    print(f"Total lines: {total_lines}")
    print(f"Lines remapped (2->1): {total_changed}")


if __name__ == "__main__":
    main()
