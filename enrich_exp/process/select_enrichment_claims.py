#!/usr/bin/env python3
"""Select target claims that should be used to enrich the EN biography."""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENRICH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(ENRICH_DIR)
DEFAULT_INPUT_JSONL = os.path.join(
    PROJECT_DIR,
    "results",
    "enrich_exp",
    "target_en_enrichment",
    "target_en_claim_candidate_verify_compact.jsonl",
)
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "enrich_exp", "target_en_enrichment")
DEFAULT_OUTPUT_JSONL = "target_claims_for_enrichment.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select target-language claims that should enrich the EN biography based on "
            "verified target-to-EN claim relations."
        )
    )
    parser.add_argument("--input-jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-jsonl", type=str, default=DEFAULT_OUTPUT_JSONL)
    return parser.parse_args()


def resolve_output_path(results_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(results_dir, path)


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_relation_index(value: Any) -> Dict[str, List[int]]:
    relation_index = value if isinstance(value, dict) else {}
    return {
        "A=B": list(relation_index.get("A=B", []) or []),
        "A>B": list(relation_index.get("A>B", []) or []),
        "B>A": list(relation_index.get("B>A", []) or []),
        "A<>B": list(relation_index.get("A<>B", []) or []),
        "contradicted": list(relation_index.get("contradicted", []) or []),
        "not_relevant": list(relation_index.get("not_relevant", []) or []),
    }


def decide_enrichment(relation_index: Dict[str, List[int]]) -> Tuple[bool, str]:
    if relation_index["A=B"]:
        return False, "covered_by_equal_en_claim"
    if relation_index["B>A"]:
        return False, "covered_by_more_informative_en_claim"
    if relation_index["A>B"] or relation_index["A<>B"]:
        return True, "target_adds_information"
    if not (
        relation_index["A=B"]
        or relation_index["A>B"]
        or relation_index["B>A"]
        or relation_index["A<>B"]
    ):
        return True, "no_aligned_en_claim"
    return False, "fallback_skip"


def build_outputs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for row in rows:
        enrich_target_claims: List[str] = []
        a_eq_b_claim_count = 0
        b_gt_a_claim_count = 0
        a_gt_b_or_a_neq_b_claim_count = 0
        other_enrich_claim_count = 0
        for target_item in row.get("target_to_en_verifications", []) or []:
            relation_index = normalize_relation_index(target_item.get("relation_en_claim_idxs"))
            if relation_index["A=B"]:
                a_eq_b_claim_count += 1
            elif relation_index["B>A"]:
                b_gt_a_claim_count += 1
            needs_enrichment, _decision_reason = decide_enrichment(relation_index)
            if needs_enrichment:
                enrich_target_claims.append(target_item.get("target_claim", ""))
                if relation_index["A>B"] or relation_index["A<>B"]:
                    a_gt_b_or_a_neq_b_claim_count += 1
                else:
                    other_enrich_claim_count += 1
        outputs.append(
            {
                "sample_idx": row.get("sample_idx"),
                "qid": row.get("qid"),
                "meta": row.get("meta", {}),
                "en_claim_count": row.get("en_claim_count"),
                "target_claim_count": row.get("target_claim_count"),
                "A_eq_B_claim_count": a_eq_b_claim_count,
                "B_gt_A_claim_count": b_gt_a_claim_count,
                "A_gt_B_or_A_neq_B_claim_count": a_gt_b_or_a_neq_b_claim_count,
                "other_enrich_claim_count": other_enrich_claim_count,
                "enrich_claim_count": len(enrich_target_claims),
                "claims": enrich_target_claims,
            }
        )
    return outputs


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    outputs = build_outputs(rows)
    output_jsonl = resolve_output_path(args.results_dir, args.output_jsonl)
    write_jsonl(output_jsonl, outputs)
    print(f"Saved enrichment selections: {output_jsonl}")


if __name__ == "__main__":
    main()
