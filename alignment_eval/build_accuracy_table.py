#!/usr/bin/env python3
"""Build per-label accuracy tables from claim alignment confusion matrices."""

import argparse
import json
import os
from typing import Any, Dict, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "scripts" else SCRIPT_DIR
REPO_DIR = os.path.dirname(EVAL_DIR)
DEFAULT_RESULTS_DIR = os.path.join(REPO_DIR, "results", "alignment_eval")
DEFAULT_OUTPUT_JSON = os.path.join(DEFAULT_RESULTS_DIR, "accuracy_table.json")

LABEL_MAP = {
    "A=B": "aligned",
    "A>B": "partial aligned (A>B)",
    "B>A": "partial aligned (B>A)",
    "A<>B": "partial aligned (A<>B)",
    "contradicted": "contradicted",
    "not relevant": "not relevant",
}
BASE_COLUMNS = ["A=B", "A>B", "B>A", "A<>B", "contradicted", "not relevant"]
SUMMARY_COLUMNS = ["ARC", "Align-FG", "Overall"]
VALUE_COLUMNS = BASE_COLUMNS + SUMMARY_COLUMNS
TEMPLATES = ["claim_only", "claim_infobox"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-label accuracy table from result confusion matrices.")
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def resolve_path(path: str, base_dir: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def mean(values: List[Optional[float]]) -> Optional[float]:
    valid = [value for value in values if value is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def find_metrics_files(results_dir: str) -> List[str]:
    paths: List[str] = []
    for root, _, files in os.walk(results_dir):
        if "claim_alignment_metrics.json" in files:
            paths.append(os.path.join(root, "claim_alignment_metrics.json"))
    return sorted(paths)


def infer_model_template(metrics_path: str, results_dir: str) -> Dict[str, str]:
    rel = os.path.relpath(metrics_path, results_dir)
    parts = rel.split(os.sep)
    return {
        "model": parts[0] if len(parts) > 0 else "",
        "template": parts[1] if len(parts) > 1 else "",
        "result_id": os.path.dirname(rel),
    }


def label_accuracy(confusion: Dict[str, Dict[str, int]], label: str) -> Optional[float]:
    row = confusion.get(label)
    if not isinstance(row, dict):
        return None
    total = sum(int(value) for value in row.values())
    correct = int(row.get(label, 0))
    return safe_div(correct, total)


def compute_values(metrics: Dict[str, Any]) -> Dict[str, Optional[float]]:
    confusion = metrics.get("confusion_matrix")
    if not isinstance(confusion, dict):
        raise ValueError("Missing or invalid confusion_matrix")

    values: Dict[str, Optional[float]] = {}
    for column in BASE_COLUMNS:
        values[column] = label_accuracy(confusion, LABEL_MAP[column])

    align_fg = mean([values["A=B"], values["A>B"], values["B>A"], values["A<>B"]])
    values["ARC"] = mean([align_fg, values["contradicted"], values["not relevant"]])
    values["Align-FG"] = align_fg
    values["Overall"] = mean([values[column] for column in BASE_COLUMNS])
    return values


def main() -> None:
    args = parse_args()
    results_dir = resolve_path(args.results_dir, EVAL_DIR)
    output_json = resolve_path(args.output_json, results_dir)
    long_rows: List[Dict[str, Any]] = []
    by_model: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}

    for metrics_path in find_metrics_files(results_dir):
        meta = infer_model_template(metrics_path, results_dir)
        if meta["template"] not in TEMPLATES:
            continue
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        values = compute_values(metrics)
        long_rows.append(
            {
                **meta,
                "metrics_path": os.path.relpath(metrics_path, EVAL_DIR),
                **values,
            }
        )
        by_model.setdefault(meta["model"], {})[meta["template"]] = values

    wide_rows: List[Dict[str, Any]] = []
    for model in sorted(by_model):
        row: Dict[str, Any] = {"model": model}
        for column in VALUE_COLUMNS:
            for template in TEMPLATES:
                values = by_model[model].get(template, {})
                row[f"{column}_{template}"] = values.get(column)
        wide_rows.append(row)

    output = {
        "definition": {
            "cell_value": "per-label accuracy = confusion_matrix[gold_label][gold_label] / sum(confusion_matrix[gold_label].values())",
            "columns": {
                "A=B": "aligned",
                "A>B": "partial aligned (A>B)",
                "B>A": "partial aligned (B>A)",
                "A<>B": "partial aligned (A<>B)",
                "contradicted": "contradicted",
                "not relevant": "not relevant",
                "ARC": "mean(Align-FG, contradicted, not relevant), where Align-FG = mean(A=B, A>B, B>A, A<>B)",
                "Align-FG": "mean(A=B, A>B, B>A, A<>B)",
                "Overall": "mean(A=B, A>B, B>A, A<>B, contradicted, not relevant)",
            },
        },
        "long_rows": long_rows,
        "wide_rows": wide_rows,
    }

    ensure_parent_dir(output_json)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote JSON: {output_json}")
    print(f"Models: {len(wide_rows)}")


if __name__ == "__main__":
    main()
