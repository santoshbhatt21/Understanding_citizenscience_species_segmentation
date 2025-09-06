import os
import re

ROOT = r"E:\Santosh_master_thesis\DATA_YOLO11_LT\labels"
ALLOWED = {0, 1}


def main():
    bad = []
    for dirpath, _, files in os.walk(ROOT):
        for fn in files:
            if not fn.lower().endswith('.txt'):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, 'r', encoding='utf-8') as fh:
                for i, ln in enumerate(fh, 1):
                    s = ln.strip()
                    if not s:
                        continue
                    m = re.match(r"^(\d+)\b", s)
                    if not m:
                        continue
                    cid = int(m.group(1))
                    if cid not in ALLOWED:
                        bad.append((p, i, cid))
                        if len(bad) >= 20:
                            break
            if len(bad) >= 20:
                break
        if len(bad) >= 20:
            break

    if bad:
        print("Found unexpected class IDs:")
        for p, i, cid in bad:
            print(f"{p}:{i} -> {cid}")
        raise SystemExit(1)
    else:
        print("OK: All labels use only class IDs 0 or 1")


if __name__ == "__main__":
    main()
