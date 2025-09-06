#!/usr/bin/env python3
r"""
View JSON metrics files from YOLO runs.
- Point it at a run folder (or any directory) to list and pretty-print *.json files.
- Useful for files like metrics_extra.json created after validation.

Examples (PowerShell):
  python .\Understanding_citizenscience_species_segmentation\script\Metrics\view_json_metrics.py -p E:\Santosh_master_thesis\segmentation_project_0\yolo11s_seg_896_LT_two_classes
  python .\Understanding_citizenscience_species_segmentation\script\Metrics\view_json_metrics.py -p E:\Santosh_master_thesis\segmentation_project_0 -r -k metrics/precision\(B\) metrics/recall\(B\)
"""
import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List


def iter_json_files(paths: List[Path], recursive: bool) -> Iterable[Path]:
    for p in paths:
        if p.is_file() and p.suffix.lower() == ".json":
            yield p
        elif p.is_dir():
            if recursive:
                for fp in p.rglob("*.json"):
                    yield fp
            else:
                for fp in p.glob("*.json"):
                    yield fp


def load_json(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e)}


def filter_keys(d: dict, keys: List[str]):
    if not isinstance(d, dict):
        return d
    if not keys:
        return d
    out = {}
    for k in keys:
        if k in d:
            out[k] = d[k]
    return out if out else d


def main():
    ap = argparse.ArgumentParser(
        description="View JSON metrics in a folder or files.")
    ap.add_argument("-p", "--paths", nargs="+",
                    default=["."], help="Paths to folders or JSON files.")
    ap.add_argument("-r", "--recursive", action="store_true",
                    help="Recurse into subfolders.")
    ap.add_argument("-k", "--keys", nargs="*", default=[],
                    help="Show only these top-level keys if present.")
    ap.add_argument("--full", action="store_true",
                    help="Print full JSON (no key filtering, pretty-printed).")
    args = ap.parse_args()

    targets = [Path(p) for p in args.paths]
    files = list(iter_json_files(targets, args.recursive))
    if not files:
        print("No JSON files found.")
        return

    for fp in sorted(files):
        try:
            size_kb = fp.stat().st_size / 1024.0
        except Exception:
            size_kb = 0.0
        print(f"\n=== {fp} ({size_kb:.1f} KB) ===")
        data = load_json(fp)
        if args.full:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            if isinstance(data, dict):
                view = data if not args.keys else filter_keys(data, args.keys)
                # Show a compact one-line and pretty block
                try:
                    print(json.dumps(view, ensure_ascii=False))
                except Exception:
                    print(str(view))
                print(json.dumps(view, indent=2, ensure_ascii=False))
            else:
                # Not a dict (e.g., list) – print length and a preview
                if isinstance(data, list):
                    print(f"List with {len(data)} items. Showing first 3:")
                    print(json.dumps(data[:3], indent=2, ensure_ascii=False))
                else:
                    print(str(data))


if __name__ == "__main__":
    main()
