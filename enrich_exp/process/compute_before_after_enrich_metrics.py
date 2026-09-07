#!/usr/bin/env python3
"""Compute before/after enrichment alignment metrics from verifier outputs."""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENRICH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(ENRICH_DIR)
if ENRICH_DIR not in sys.path:
    sys.path.insert(0, ENRICH_DIR)

from utils import display_path_from_repo, display_paths_from_repo

DEFAULT_RESULTS_ROOT = os.path.join(PROJECT_DIR, "results", "enrich_exp", "before_after_enrich_verify")
DEFAULT_INPUT_GLOB = os.path.join(
    DEFAULT_RESULTS_ROOT,
    "*",
    "*",
    "*",
    "before_after_enrich_verify_compact.jsonl",
)
DEFAULT_OUTPUT_JSON = os.path.join(DEFAULT_RESULTS_ROOT, "before_after_enrich_metrics_summary.json")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute before-support quality metrics from before/after enrichment "
            "verifier compact JSONL files."
        )
    )
    parser.add_argument("--results-root", type=str, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument(
        "--input-jsonl",
        nargs="+",
        default=None,
        help=(
            "One or more compact verifier JSONL files or glob patterns. "
            "Default: scan results-root/*/*/*/before_after_enrich_verify_compact.jsonl."
        ),
    )
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def discover_input_jsonl(paths: Optional[Sequence[str]], results_root: str) -> List[str]:
    if paths:
        discovered: List[str] = []
        for path in paths:
            matches = sorted(glob.glob(path))
            discovered.extend(matches if matches else [path])
        return discovered

    pattern = os.path.join(
        os.path.abspath(results_root),
        "*",
        "*",
        "*",
        "before_after_enrich_verify_compact.jsonl",
    )
    discovered = sorted(glob.glob(pattern))
    if not discovered:
        raise FileNotFoundError(f"No compact verifier files found with pattern: {pattern}")
    return discovered


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def claim_count(sample: Dict[str, Any], key: str, fallback_len: int) -> int:
    value = safe_int(sample.get(key))
    if value is None and isinstance(sample.get("counts"), dict):
        value = safe_int(sample["counts"].get(key))
    if value is None:
        return fallback_len
    return max(value, fallback_len)


def normalized_enrichment(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def relation_from_candidate(candidate: Dict[str, Any]) -> str:
    relation = str(candidate.get("relation", "")).strip()
    if relation:
        return relation

    parsed = candidate.get("parsed_response")
    if not isinstance(parsed, dict):
        return "not_relevant"

    alignment = str(parsed.get("Alignment", "")).strip()
    enrichment = normalized_enrichment(parsed.get("Enrichment"))
    if alignment == "Aligned":
        if enrichment == "" or enrichment.casefold() == "none":
            return "A=B"
        if enrichment == "A > B":
            return "A>B"
        if enrichment == "B > A":
            return "B>A"
        if enrichment == "A <> B":
            return "A<>B"
        return "A<>B"
    if alignment == "Contradicted":
        return "contradicted"
    return "not_relevant"


def direction_relation_counts(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = {
        "A=B": 0,
        "A>B": 0,
        "B>A": 0,
        "A<>B": 0,
        "contradicted": 0,
        "not_relevant": 0,
    }
    for item in items:
        for candidate in item.get("candidates", []) or []:
            relation = relation_from_candidate(candidate)
            counts[relation] = counts.get(relation, 0) + 1
    return counts


def item_relation_set(item: Dict[str, Any]) -> set:
    return {
        relation_from_candidate(candidate)
        for candidate in item.get("candidates", []) or []
    }


def invert_ab_relation(relation: str) -> str:
    if relation == "A>B":
        return "B>A"
    if relation == "B>A":
        return "A>B"
    return relation


def after_item_relation_set_as_before_after(item: Dict[str, Any]) -> set:
    """Return after_to_before relations with A=before_enrich and B=after_enrich."""
    return {
        invert_ab_relation(relation)
        for relation in item_relation_set(item)
    }


def summarize_after_quality_item(item: Dict[str, Any]) -> Dict[str, int]:
    relations = after_item_relation_set_as_before_after(item)
    counts = {
        "valid_n": 0,
        "halluc_n": 0,
        "new_info_n": 0,
        "contrad_n": 0,
        "not_rel_n": 0,
        "unknown_n": 0,
    }

    if {"A=B", "A>B"} & relations:
        counts["valid_n"] = 1
    elif {"B>A", "A<>B"} & relations:
        counts["halluc_n"] = 1
        counts["new_info_n"] = 1
    elif "contradicted" in relations:
        counts["halluc_n"] = 1
        counts["contrad_n"] = 1
    elif "not_relevant" in relations:
        counts["halluc_n"] = 1
        counts["not_rel_n"] = 1
    else:
        counts["halluc_n"] = 1
        counts["unknown_n"] = 1
    return counts


def summarize_after_quality(
    after_items: Sequence[Dict[str, Any]],
    after_count: int,
) -> Dict[str, Any]:
    counts = {
        "valid_n": 0,
        "halluc_n": 0,
        "new_info_n": 0,
        "contrad_n": 0,
        "not_rel_n": 0,
        "unknown_n": 0,
    }
    for item in after_items:
        item_counts = summarize_after_quality_item(item)
        for key, value in item_counts.items():
            counts[key] += value

    counted_primary = counts["valid_n"] + counts["halluc_n"]
    if after_count > counted_primary:
        missing_n = after_count - counted_primary
        counts["halluc_n"] += missing_n
        counts["unknown_n"] += missing_n

    output = dict(counts)
    output["valid_rate"] = counts["valid_n"] / after_count if after_count else 0.0
    output["halluc_rate"] = counts["halluc_n"] / after_count if after_count else 0.0
    output["new_info_rate"] = counts["new_info_n"] / after_count if after_count else 0.0
    output["contrad_rate"] = counts["contrad_n"] / after_count if after_count else 0.0
    output["not_rel_rate"] = counts["not_rel_n"] / after_count if after_count else 0.0
    output["unknown_rate"] = counts["unknown_n"] / after_count if after_count else 0.0
    return output


def summarize_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    before_items = sample.get("before_to_after_verifications", []) or []
    after_items = sample.get("after_to_before_verifications", []) or []
    before_en_count = claim_count(sample, "before_en_claim_count", 0)
    before_target_count = claim_count(sample, "before_target_claim_count", 0)
    before_total_count = claim_count(sample, "before_claim_count", len(before_items))
    after_count = claim_count(sample, "after_claim_count", len(after_items))
    quality = summarize_after_quality(after_items, after_count)

    return {
        "input_idx": sample.get("input_idx"),
        "qid": sample.get("qid"),
        "name": sample.get("name") or sample.get("meta", {}).get("name"),
        "target_lang": sample.get("target_lang") or sample.get("meta", {}).get("target_lang"),
        "before_en_n": before_en_count,
        "before_target_n": before_target_count,
        "before_total_n": before_total_count,
        "after_n": after_count,
        "b2a_rel": direction_relation_counts(before_items),
        "a2b_rel": direction_relation_counts(after_items),
        "quality": quality,
    }


def aggregate_sample_macro(sample_summaries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    new_info_rate = mean([sample["quality"]["new_info_rate"] for sample in sample_summaries])
    contrad_rate = mean([sample["quality"]["contrad_rate"] for sample in sample_summaries])
    not_rel_rate = mean([sample["quality"]["not_rel_rate"] for sample in sample_summaries])
    unknown_rate = mean([sample["quality"]["unknown_rate"] for sample in sample_summaries])
    before_en_n = mean([sample["before_en_n"] for sample in sample_summaries])
    before_total_n = mean([sample["before_total_n"] for sample in sample_summaries])
    after_n = mean([sample["after_n"] for sample in sample_summaries])
    return {
        "sample_count": len(sample_summaries),
        "before_en_n": before_en_n,
        "before_target_n": mean([sample["before_target_n"] for sample in sample_summaries]),
        "before_total_n": before_total_n,
        "after_n": after_n,
        "after_en_delta": after_n - before_en_n,
        "after_before_ratio": after_n / before_total_n if before_total_n else 0.0,
        "valid_n": mean([sample["quality"]["valid_n"] for sample in sample_summaries]),
        "halluc_n": mean([sample["quality"]["halluc_n"] for sample in sample_summaries]),
        "valid_rate": mean([sample["quality"]["valid_rate"] for sample in sample_summaries]),
        "halluc_rate": mean([sample["quality"]["halluc_rate"] for sample in sample_summaries]),
        "hallucination": {
            "new_info_rate": new_info_rate,
            "contrad_rate": contrad_rate,
            "not_rel_rate": not_rel_rate,
            "unknown_rate": unknown_rate,
        },
    }


def add_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:
    output = dict(left)
    for key, value in right.items():
        output[key] = output.get(key, 0) + value
    return output


def aggregate_relation_counts(sample_summaries: Sequence[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for sample in sample_summaries:
        counts = add_counts(counts, sample.get(key, {}))
    return counts


def infer_run_labels(path: str, results_root: str) -> Tuple[str, str, str]:
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(results_root)
    try:
        rel_path = os.path.relpath(abs_path, abs_root)
    except ValueError:
        rel_path = abs_path
    parts = rel_path.split(os.sep)
    if len(parts) >= 4:
        return parts[0], parts[1], parts[2]
    return "unknown_method", "unknown_lang", "unknown_model"


def display_input_path(path: str, results_root: str) -> str:
    return display_path_from_repo(path)


def summarize_file(path: str, results_root: str) -> Dict[str, Any]:
    method, target_lang, model_name = infer_run_labels(path, results_root)
    samples = [summarize_sample(sample) for sample in read_jsonl(path)]
    if not samples:
        raise ValueError(f"No samples loaded from {path}")
    summary = aggregate_sample_macro(samples)
    return {
        "method": method,
        "target_lang": target_lang,
        "model_name": model_name,
        "input_jsonl": display_input_path(path, results_root),
        **summary,
        "rel_counts": {
            "b2a": aggregate_relation_counts(samples, "b2a_rel"),
            "a2b": aggregate_relation_counts(samples, "a2b_rel"),
        },
    }


def main() -> None:
    args = parse_args()
    input_jsonl = discover_input_jsonl(args.input_jsonl, args.results_root)
    runs = [summarize_file(path, args.results_root) for path in input_jsonl]
    summary = {
        "results_root": display_path_from_repo(args.results_root),
        "input_jsonl": display_paths_from_repo(input_jsonl),
        "run_count": len(runs),
        "runs": runs,
        "metric_note": (
            "Metrics evaluate whether after_enrich claims are supported by before_enrich claims. "
            "All rates are sample-macro averages over after_to_before_verifications. "
            "For metric computation, relation labels are interpreted as A=before_enrich and B=after_enrich. "
            "The compact after_to_before labels are inverted before scoring because they are stored with "
            "A=after_enrich and B=before_enrich. valid_n/valid_rate count after claims with B=A or B<A "
            "(equivalently A=B or A>B under A=before, B=after). halluc_n/halluc_rate count after claims "
            "that do not satisfy this support condition. Hallucination categories are mutually exclusive "
            "with priority: new_info_rate for B>A or A<>B, contrad_rate for contradicted when no aligned "
            "relation exists, not_rel_rate for not_relevant when no aligned/contradicted relation exists, "
            "and unknown_rate for claims without a classified supporting candidate."
        ),
    }
    write_json(args.output_json, summary)
    print(f"Loaded compact verifier files: {len(input_jsonl)}")
    for path in input_jsonl:
        print(f"  {path}")
    print(f"Saved JSON metrics: {args.output_json}")


if __name__ == "__main__":
    main()
