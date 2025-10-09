#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Threshold sweep utility for pseudo-labeling.

Usage example:
python sweep_thresholds_from_predictions.py \
  --pred-csv "E:/Santosh_master_thesis/prediction_metadata_Leaves_Others_Trunks.csv" \
  --gt-csv   "E:/Santosh_master_thesis/val_ground_truth.csv" \
  --out-prefix "E:/Santosh_master_thesis/threshold_sweep_LOT" \
  --target-precision 0.98 --min-coverage 50

Ground truth CSV format:
    image_path,true_class
    E:/.../img001.jpg,Leaves
    E:/.../img002.jpg,Others
    ...

Notes:
- We join predictions and GT by exact image_path first; rows that don't match are then
  attempted by basename filename matching. Unmatched rows are dropped with a warning count.
- We compute precision@threshold for each class c using rows where predicted_class==c,
  and choose the **smallest** threshold with precision >= target_precision.
- Coverage reported as #accepted / #pred_as_class (within the eval subset).
"""

import os
import json
import argparse
import math
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

# Make plotting optional; don't crash if matplotlib isn't installed
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Sweep confidence thresholds and suggest per-class/global cutoffs.")
    ap.add_argument("--pred-csv", required=True,
                    help="Predictions CSV from the CLI (needs predicted_class, confidence).")
    ap.add_argument("--gt-csv", required=True,
                    help="CSV with columns image_path,true_class.")
    ap.add_argument("--out-prefix", required=True,
                    help="Prefix path for outputs (no extension).")
    ap.add_argument("--target-precision", type=float,
                    default=0.98, help="Target precision per class.")
    ap.add_argument("--min-coverage", type=int, default=0,
                    help="Minimum accepted count at the chosen threshold (0 to ignore).")
    ap.add_argument("--thr-start", type=float, default=0.90,
                    help="Start of threshold sweep (inclusive).")
    ap.add_argument("--thr-stop", type=float, default=0.999,
                    help="End of threshold sweep (inclusive).")
    ap.add_argument("--thr-step", type=float,
                    default=0.0025, help="Threshold step.")
    # Column overrides (for flexibility with different CSV schemas)
    ap.add_argument("--pred-image-col", default="image_path",
                    help="Column name in pred CSV for image path")
    ap.add_argument("--pred-class-col", default="predicted_class",
                    help="Column name in pred CSV for predicted class")
    ap.add_argument("--pred-conf-col", default="confidence",
                    help="Column name in pred CSV for confidence score")
    ap.add_argument("--gt-image-col", default="image_path",
                    help="Column name in GT CSV for image path")
    ap.add_argument("--gt-class-col", default="true_class",
                    help="Column name in GT CSV for ground-truth class")
    ap.add_argument("--no-plots", action="store_true",
                    help="Disable plotting precision curves")
    return ap.parse_args()


def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _find_first_present(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise KeyError(
        f"None of the candidate columns present: {candidates}. Available: {list(df.columns)}")


def _normalize_pred_columns(df: pd.DataFrame, img_col: str, cls_col: str, conf_col: str) -> pd.DataFrame:
    # Accept common synonyms if user-provided columns are missing
    img = img_col if img_col in df.columns else _find_first_present(
        df, (img_col, "image", "img_path", "filepath", "path", "file"))
    cls = cls_col if cls_col in df.columns else _find_first_present(
        df, (cls_col, "label", "class", "pred_class", "pred"))
    conf = conf_col if conf_col in df.columns else _find_first_present(
        df, (conf_col, "score", "prob", "confidence_score"))
    out = df.rename(
        columns={img: "image_path", cls: "predicted_class", conf: "confidence"}).copy()
    return out


def _normalize_gt_columns(df: pd.DataFrame, img_col: str, cls_col: str) -> pd.DataFrame:
    img = img_col if img_col in df.columns else _find_first_present(
        df, (img_col, "image", "img_path", "filepath", "path", "file"))
    cls = cls_col if cls_col in df.columns else _find_first_present(
        df, (cls_col, "label", "class", "gt_class", "true_label"))
    out = df.rename(columns={img: "image_path", cls: "true_class"}).copy()
    return out


def _merge_pred_gt(pred: pd.DataFrame, gt: pd.DataFrame) -> pd.DataFrame:
    # First try exact path
    m = pred.merge(gt, on="image_path", how="inner")
    if len(m) >= 0.5 * len(gt):
        return m
    # Fallback: basename match
    pred2 = pred.copy()
    gt2 = gt.copy()
    pred2["__base"] = pred2["image_path"].apply(
        lambda p: os.path.basename(str(p)))
    gt2["__base"] = gt2["image_path"].apply(lambda p: os.path.basename(str(p)))
    m2 = pred2.merge(gt2[["__base", "true_class"]],
                     on="__base", how="inner").drop(columns=["__base"])
    # Prefer the better match set
    return m if len(m) >= len(m2) else m2


def _precision_at_threshold(df_cls: pd.DataFrame, thr: float) -> Tuple[float, int, int]:
    """Return (precision, accepted, total) within rows for one class (predicted_class==c)."""
    if df_cls.empty:
        return float("nan"), 0, 0
    sel = df_cls["confidence"] >= thr
    acc = (df_cls.loc[sel, "is_correct"].sum()) if sel.any() else 0
    tot = int(sel.sum())
    if tot == 0:
        return float("nan"), 0, int(len(df_cls))
    prec = float(acc) / float(tot)
    return prec, int(tot), int(len(df_cls))


def _plot_precision_curve(thrs: List[float], precs: List[float], cls_name: str, out_path: str):
    plt.figure(figsize=(6, 4))
    plt.plot(thrs, precs)
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title(f"Precision vs Threshold - {cls_name}")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    args = _parse_args()
    _ensure_dir(args.out_prefix)

    pred_raw = pd.read_csv(args.pred_csv)
    gt_raw = pd.read_csv(args.gt_csv)

    # Normalize/rename columns to canonical names
    try:
        pred = _normalize_pred_columns(
            pred_raw, args.pred_image_col, args.pred_class_col, args.pred_conf_col)
    except Exception as e:
        raise ValueError(
            f"Failed to locate prediction columns in {args.pred_csv}: {e}")
    try:
        gt = _normalize_gt_columns(
            gt_raw, args.gt_image_col, args.gt_class_col)
    except Exception as e:
        raise ValueError(
            f"Failed to locate ground-truth columns in {args.gt_csv}: {e}")

    # Merge
    m = _merge_pred_gt(pred, gt)
    if m.empty:
        raise RuntimeError(
            "No overlap between predictions and ground truth. Check paths/filenames.")
    unmatched = len(gt) - len(m)
    if unmatched > 0:
        print(
            f"[Warn] Dropped {unmatched} GT rows without matching predictions after merge.")

    # Compute correctness
    m["is_correct"] = (m["predicted_class"].astype(
        str) == m["true_class"].astype(str)).astype(int)

    classes = sorted(m["predicted_class"].astype(str).unique().tolist())
    thrs = np.arange(args.thr_start, args.thr_stop +
                     1e-9, args.thr_step).tolist()

    # Sweep per class
    rows = []
    recommended: Dict[str, float] = {}
    for c in classes:
        df_c = m[m["predicted_class"] == c].copy()
        if df_c.empty:
            continue
        precs, covs = [], []
        best_tau = None
        for t in thrs:
            p, accepted, total = _precision_at_threshold(df_c, t)
            precs.append(p if not math.isnan(p) else np.nan)
            covs.append(accepted)
            ok_prec = (not math.isnan(p)) and (p >= args.target_precision)
            ok_cov = (
                accepted >= args.min_coverage) if args.min_coverage > 0 else True
            if (best_tau is None) and ok_prec and ok_cov:
                best_tau = float(t)

            rows.append({"class": c, "threshold": float(t), "precision": None if math.isnan(p) else float(p),
                         "accepted": int(accepted), "total_pred_as_class": int(total)})
        # Recommend
        if best_tau is None:
            # Fallback: choose the smallest tau giving the highest precision achievable
            try:
                valid = [(t, p)
                         for t, p in zip(thrs, precs) if not math.isnan(p)]
                if valid:
                    max_prec = max(p for _, p in valid)
                    best_tau = min(t for t, p in valid if p == max_prec)
                else:
                    best_tau = 0.98
            except Exception:
                best_tau = 0.98
        recommended[c] = float(best_tau)

        # Plot curve
        if (plt is not None) and (not args.no_plots):
            try:
                _plot_precision_curve(thrs, [0.0 if math.isnan(p) else p for p in precs],
                                      c, f"{args.out_prefix}_precision_curve_{c}.png")
            except Exception as e:
                print(f"[Warn] Plot failed for class {c}: {e}")

    # Global suggestion:
    # choose the minimum over per-class recommended to ensure ALL pass the target precision
    global_min = float(min(recommended.values())) if recommended else 0.98

    # Save long sweep table
    sweep_df = pd.DataFrame(rows)
    sweep_csv = f"{args.out_prefix}_sweep_per_class.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    # Save recommendations
    rec_json = {
        "target_precision": args.target_precision,
        "min_coverage": args.min_coverage,
        "GLOBAL_MIN": global_min,
        "PER_CLASS_MIN": recommended,
        "notes": "GLOBAL_MIN is the min of per-class thresholds so that all classes reach target precision; "
                 "you can override specific classes with PER_CLASS_MIN and keep a looser GLOBAL_MIN if desired."
    }
    rec_path = f"{args.out_prefix}_recommended.json"
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(rec_json, f, indent=2)

    # Human summary
    lines = [f"Target precision: {args.target_precision} | Min coverage: {args.min_coverage}",
             f"Suggested GLOBAL_MIN (all classes ≥ target): {global_min:.4f}",
             "Per-class suggestions:"]
    for c in classes:
        tau = recommended.get(c, None)
        lines.append(
            f"  - {c}: {tau:.4f}" if tau is not None else f"  - {c}: N/A")
    summary_txt = "\n".join(lines)
    with open(f"{args.out_prefix}_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)

    print("\n=== Threshold Sweep Complete ===")
    print(summary_txt)
    print(
        f"\nFiles written:\n- {sweep_csv}\n- {rec_path}\n- {args.out_prefix}_summary.txt")
    print(f"- {args.out_prefix}_precision_curve_<class>.png (one per class)")


if __name__ == "__main__":
    main()
#!/usr/bin/env python
