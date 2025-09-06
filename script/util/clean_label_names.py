import os
import re
import json
import argparse


def clean_label(s: str) -> str:
    s = re.sub(r"^\s*\d+[_\s]+", "", s)
    s = s.replace("_", " ")
    return s


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(
        description='Create cleaned-label copies of Training_Stats JSONs (non-destructive).')
    p.add_argument('--ckpt-root', type=str, required=True,
                   help='Checkpoint root containing Training_Stats')
    p.add_argument('--out-suffix', type=str, default='_clean',
                   help='Suffix inserted before .json for outputs')
    args = p.parse_args()

    stats = os.path.join(args.ckpt_root, 'Training_Stats')
    if not os.path.isdir(stats):
        raise FileNotFoundError(
            f"Training_Stats not found in {args.ckpt_root}")

    # 1) summary.json -> summary_clean.json (clean keys in class_to_idx)
    sum_path = os.path.join(stats, 'summary.json')
    if os.path.exists(sum_path):
        summary = load_json(sum_path)
        cti = summary.get('class_to_idx') or {}
        cleaned = {clean_label(k): int(v) for k, v in cti.items()}
        out = dict(summary)
        out['class_to_idx'] = cleaned
        save_json(os.path.join(stats, f'summary{args.out_suffix}.json'), out)

        # Also write label order text for convenience
        idx_to_label = {int(v): k for k, v in cleaned.items()}
        labels = [idx_to_label[i] for i in sorted(idx_to_label)]
        with open(os.path.join(stats, f'labels{args.out_suffix}.txt'), 'w', encoding='utf-8') as f:
            for lab in labels:
                f.write(lab + '\n')

    # 2) classification_report.json -> classification_report_clean.json (rename per-class keys only)
    rep_path = os.path.join(stats, 'classification_report.json')
    if os.path.exists(rep_path):
        report = load_json(rep_path)
        out = {}
        for k, v in report.items():
            if isinstance(v, dict) and k not in {'accuracy', 'macro avg', 'weighted avg'}:
                out[clean_label(k)] = v
            else:
                out[k] = v
        save_json(os.path.join(
            stats, f'classification_report{args.out_suffix}.json'), out)

    # 3) confusion_matrix.json -> confusion_matrix_clean_labels.json (same matrix, cleaned labels)
    cm_path = os.path.join(stats, 'confusion_matrix.json')
    if os.path.exists(cm_path):
        cm = load_json(cm_path)
        labels = cm.get('labels') or []
        cm['labels'] = [clean_label(l) for l in labels]
        save_json(os.path.join(
            stats, f'confusion_matrix{args.out_suffix}.json'), cm)

    print('Done. Cleaned copies written (if sources existed).')


if __name__ == '__main__':
    main()
