import argparse
import json
from pathlib import Path


def compute_f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def load_metrics_json(run_dir: Path) -> dict:
    # Ultralytics writes metrics to metrics.json in the val/run folder
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found in {run_dir}")
    with metrics_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-class and overall F1 from YOLO val metrics.json.")
    parser.add_argument(
        "run_dir",
        type=str,
        help="Path to YOLO validation run directory (e.g., runs/segment/val or val2).",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    metrics = load_metrics_json(run_dir)

    # Expected structure: metrics should contain 'metrics' -> 'seg' or top-level keys with per-class stats
    # We try a few common layouts but keep it simple: look for a 'classes' list with dicts.
    classes = None
    if isinstance(metrics, dict):
        if "classes" in metrics:
            classes = metrics["classes"]
        elif "seg" in metrics and isinstance(metrics["seg"], dict) and "classes" in metrics["seg"]:
            classes = metrics["seg"]["classes"]

    if not classes:
        raise SystemExit(
            "Could not find per-class metrics in metrics.json (expected a 'classes' list). "
            "Open the file to inspect its structure."
        )

    print("Per-class F1 scores:")
    print("cls\tname\tP\tR\tF1\tmAP50\tmAP50-95")

    f1_values = []
    support = []  # number of objects per class if available

    for c in classes:
        # Heuristic: support can be under 'n' or 'instances'
        p = float(c.get("p", 0.0))
        r = float(c.get("r", 0.0))
        ap50 = float(c.get("ap50", 0.0))
        ap = float(c.get("ap", 0.0))  # often mAP50-95 per class
        f1 = compute_f1(p, r)

        f1_values.append(f1)
        support.append(float(c.get("n", c.get("instances", 0.0))))

        cls_id = c.get("id", c.get("cls", "?"))
        name = c.get("name", "?")

        print(
            f"{cls_id}\t{name}\t{p:.3f}\t{r:.3f}\t{f1:.3f}\t{ap50:.3f}\t{ap:.3f}"
        )

    # Overall macro F1
    if f1_values:
        macro_f1 = sum(f1_values) / len(f1_values)
    else:
        macro_f1 = 0.0

    # Overall weighted F1 (by number of instances) if counts available
    if any(s > 0 for s in support):
        total = sum(support)
        weighted_f1 = sum(f * s for f, s in zip(f1_values, support)) / total
    else:
        weighted_f1 = macro_f1

    print("\nOverall F1 scores:")
    print(f"Macro F1 (unweighted mean over classes): {macro_f1:.3f}")
    print(f"Weighted F1 (weighted by instances):   {weighted_f1:.3f}")


if __name__ == "__main__":
    main()
