import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
except Exception:
    yaml = None

# Configure default dataset YAML, can be overridden via CLI arg
DATA_YAML = r"E:/Santosh_master_thesis/DATA_YOLO11_strict_CAM_setting/data_20_classes.yaml"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def load_yaml(path: str) -> Dict:
    if yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def iter_label_files(labels_root: Path) -> List[Path]:
    return [p for p in labels_root.rglob("*.txt") if p.is_file()]


def parse_label_line(line: str) -> Tuple[int, List[float]]:
    parts = line.strip().split()
    if not parts:
        raise ValueError("empty")
    cls_id = int(float(parts[0]))
    coords = [float(x) for x in parts[1:]]
    return cls_id, coords


def check_file(lbl_path: Path) -> Dict:
    issues = {
        "empty": False,
        "bad_lines": 0,
        "out_of_range": 0,
        "too_few_points": 0,
        "instances": 0,
        "classes": set(),
        "max_vertices": 0,
    }
    try:
        text = lbl_path.read_text(encoding="utf-8").strip()
    except Exception:
        issues["bad_lines"] += 1
        return issues

    if not text:
        issues["empty"] = True
        return issues

    for line in text.splitlines():
        try:
            cls_id, coords = parse_label_line(line)
            issues["classes"].add(cls_id)
            # YOLO polygon: class + x1 y1 x2 y2 ... (2*N coords)
            if len(coords) < 6 or len(coords) % 2 == 1:
                issues["too_few_points"] += 1
                continue
            # range checks
            bad = sum(1 for v in coords if v < 0.0 or v > 1.0)
            issues["out_of_range"] += bad
            issues["instances"] += 1
            issues["max_vertices"] = max(
                issues["max_vertices"], len(coords) // 2)
        except Exception:
            issues["bad_lines"] += 1
    return issues


def main():
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else DATA_YAML
    cfg = load_yaml(yaml_path)
    base = cfg.get("path", "")
    labels_train = cfg.get("labels", {}).get(
        "train") if isinstance(cfg.get("labels"), dict) else None
    labels_val = cfg.get("labels", {}).get(
        "val") if isinstance(cfg.get("labels"), dict) else None

    # Fall back to default convention
    labels_train = labels_train or os.path.join("labels", "train")
    labels_val = labels_val or os.path.join("labels", "val")

    train_root = Path(labels_train if os.path.isabs(
        labels_train) else os.path.join(base, labels_train))
    val_root = Path(labels_val if os.path.isabs(labels_val)
                    else os.path.join(base, labels_val))

    report = {}
    for split, root in [("train", train_root), ("val", val_root)]:
        if not root.exists():
            report[split] = {"error": f"labels root not found: {root}"}
            continue
        files = iter_label_files(root)
        split_stats = {
            "files": len(files),
            "total_instances": 0,
            "total_empty": 0,
            "total_bad_lines": 0,
            "total_out_of_range": 0,
            "total_too_few_points": 0,
            "classes_present": set(),
            "max_vertices_any": 0,
            "instances_per_image": [],
        }
        for f in files:
            issues = check_file(f)
            split_stats["total_instances"] += issues["instances"]
            split_stats["total_empty"] += 1 if issues["empty"] else 0
            split_stats["total_bad_lines"] += issues["bad_lines"]
            split_stats["total_out_of_range"] += issues["out_of_range"]
            split_stats["total_too_few_points"] += issues["too_few_points"]
            split_stats["classes_present"].update(issues["classes"])
            split_stats["max_vertices_any"] = max(
                split_stats["max_vertices_any"], issues["max_vertices"])
            split_stats["instances_per_image"].append(issues["instances"])
        report[split] = split_stats

    # finalize sets for JSON
    for split in ("train", "val"):
        if split in report and isinstance(report[split], dict) and "classes_present" in report[split]:
            report[split]["classes_present"] = sorted(
                list(report[split]["classes_present"]))

    out_path = Path(__file__).with_name("yolo_seg_label_health.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
