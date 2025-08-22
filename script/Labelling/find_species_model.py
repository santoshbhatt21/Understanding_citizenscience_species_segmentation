import argparse
import os
import sys
import json
import datetime as dt
from typing import Dict, Any, Optional, Tuple

import torch


def parse_args():
    p = argparse.ArgumentParser(
        description="Find species (9-class) model checkpoints by date range and hints.")
    p.add_argument("--root", default="E:/Santosh_master_thesis",
                   help="Root directory to scan recursively")
    p.add_argument("--from", dest="from_date", default="2025-06-01",
                   help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--to", dest="to_date", default="2025-07-31",
                   help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--classes", type=int, default=9,
                   help="Target number of classes to match")
    p.add_argument("--name-hint", default="species",
                   help="Substring to filter paths (case-insensitive). Use '' to disable.")
    p.add_argument("--limit", type=int, default=50,
                   help="Max number of candidates to print")
    return p.parse_args()


def load_state_dict_info(pth_path: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Load the checkpoint file safely and return (state_dict_like, raw_obj). Handles plain state_dict
    and wrapper dicts (e.g., {'state_dict': ...}). On failure, returns (None, None).
    """
    try:
        obj = torch.load(pth_path, map_location="cpu")
    except Exception:
        return None, None

    if isinstance(obj, dict):
        # PyTorch Lightning often stores 'state_dict'
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"], obj
        # Plain state dict
        if all(isinstance(k, str) for k in obj.keys()):
            return obj, obj
    return None, obj


def infer_num_classes(state_dict: Dict[str, Any]) -> Optional[int]:
    """Try to infer number of output classes from classifier/fc/head weight shape."""
    if state_dict is None:
        return None
    # Candidate classifier keys to inspect
    candidate_keys = [
        k for k in state_dict.keys() if k.endswith(".weight") and any(
            part in k.lower() for part in ["classifier", "fc", "head", "output", "logits"]
        )
    ]
    best_guess = None
    for k in candidate_keys:
        w = state_dict.get(k)
        try:
            if hasattr(w, "shape") and len(w.shape) == 2:
                out_ch = int(w.shape[0])
                # Heuristic bounds: ignore too-large heads (e.g., > 200)
                if 1 <= out_ch <= 200:
                    best_guess = out_ch
                    # Prefer EfficientNet-likelies
                    if "classifier" in k.lower():
                        return out_ch
        except Exception:
            continue
    return best_guess


def try_load_class_report(ckpt_dir: str) -> Optional[Dict[str, Any]]:
    stats = os.path.join(ckpt_dir, "Training_Stats",
                         "classification_report.json")
    if os.path.isfile(stats):
        try:
            with open(stats, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def within_range(path: str, start: dt.datetime, end: dt.datetime) -> bool:
    try:
        mtime = dt.datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        return False
    return start <= mtime <= end


def main():
    args = parse_args()
    start = dt.datetime.fromisoformat(args.from_date)
    end = dt.datetime.fromisoformat(
        args.to_date) + dt.timedelta(days=1) - dt.timedelta(seconds=1)
    name_hint = args.name_hint.lower() if args.name_hint else ""

    candidates = []
    for root, _, files in os.walk(args.root):
        # Optional name hint on path
        if name_hint and name_hint not in root.lower():
            # Skip directories that don't include the hint
            continue
        for fn in files:
            if not fn.lower().endswith(".pth"):
                continue
            # Prefer best_model_* but don't require it
            if "best_model" not in fn.lower() and name_hint:
                continue
            p = os.path.join(root, fn)
            if not within_range(p, start, end):
                continue
            candidates.append(p)

    if not candidates:
        print("No checkpoint candidates found in date range.")
        print(
            f"Scanned root={args.root} range=[{args.from_date}..{args.to_date}] hint='{args.name_hint}'")
        sys.exit(0)

    print(f"Found {len(candidates)} candidate checkpoints. Inspecting…")
    # Sort by mtime desc (newest first)
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    printed = 0
    for p in candidates:
        if printed >= args.limit:
            break
        sd, raw = load_state_dict_info(p)
        n_classes = infer_num_classes(sd) if sd is not None else None
        cls_report = try_load_class_report(os.path.dirname(p))

        # Filter by num classes if known
        if n_classes is not None and n_classes != args.classes:
            continue
        # If num classes unknown, allow but mark as unknown

        printed += 1
        mtime = dt.datetime.fromtimestamp(
            os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print("\n=== Candidate ===")
        print(f"Path: {p}")
        print(f"Modified: {mtime} | Size: {size_mb:.1f} MB")
        print(
            f"Inferred classes: {n_classes if n_classes is not None else 'unknown'}")
        if cls_report:
            labels = [k for k in cls_report.keys() if k not in (
                "accuracy", "macro avg", "weighted avg")]
            print(f"Report labels ({len(labels)}): {labels}")
            print(f"Accuracy: {cls_report.get('accuracy', 'n/a')}")
        else:
            print("No classification_report.json found adjacent.")

    if printed == 0:
        print("No candidates matched the expected class count. Try relaxing --name-hint or remove it.")


if __name__ == "__main__":
    main()
