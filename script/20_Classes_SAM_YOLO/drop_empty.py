"""
Fast drop-empty utility:
Moves images without a non-empty matching label to _empty_quarantine, along with empty label files.

Speedups over original:
- Pre-index label files (existence + non-empty) to avoid per-image stat calls.
- Iterate only image files.
- Use os.rename (same drive) and minimal directory creation.
- Progress logging every N images.
"""

from pathlib import Path
import os
import time

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ROOT = Path(r"E:/Santosh_master_thesis/DATA_YOLO11_strict_CAM_setting")
LOG_EVERY = 100


def iter_images(root: Path):
    for ext in IMAGE_EXTS:
        for p in root.rglob(f"*{ext}"):
            if "_empty_quarantine" in p.parts:
                continue
            yield p


def main():
    t0 = time.time()
    for split in ["train", "val"]:
        imdir = ROOT / "images" / split
        lbdir = ROOT / "labels" / split
        qimg = imdir / "_empty_quarantine"
        qlbl = lbdir / "_empty_quarantine"
        qimg.mkdir(parents=True, exist_ok=True)
        qlbl.mkdir(parents=True, exist_ok=True)

        # Index label files
        existing_lbl = set()
        non_empty_lbl = set()
        for lbl in lbdir.rglob("*.txt"):
            rel = lbl.relative_to(lbdir)
            existing_lbl.add(rel)
            try:
                if lbl.stat().st_size > 0:
                    non_empty_lbl.add(rel)
            except Exception:
                pass

        n_keep = 0
        n_drop = 0
        for i, p in enumerate(iter_images(imdir), 1):
            rel = p.relative_to(imdir)
            rel_txt = rel.with_suffix(".txt")

            if rel_txt in non_empty_lbl:
                n_keep += 1
            else:
                # Ensure target dirs exist
                (qimg / rel).parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.rename(str(p), str(qimg / rel))
                except Exception:
                    # Fall back to replace (handles existing)
                    try:
                        os.replace(str(p), str(qimg / rel))
                    except Exception:
                        # Last resort, skip
                        pass

                if rel_txt in existing_lbl:
                    lbl = lbdir / rel_txt
                    (qlbl / rel_txt).parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.rename(str(lbl), str(qlbl / rel_txt))
                    except Exception:
                        try:
                            os.replace(str(lbl), str(qlbl / rel_txt))
                        except Exception:
                            pass
                n_drop += 1

            if i % LOG_EVERY == 0:
                print(
                    f"{split}: processed {i:,} | kept {n_keep:,} | moved {n_drop:,}")

        print(f"{split}: kept {n_keep:,}, moved {n_drop:,} empty.")

    dt = time.time() - t0
    print(f"Done in {dt/60:.1f} min")


if __name__ == "__main__":
    main()
