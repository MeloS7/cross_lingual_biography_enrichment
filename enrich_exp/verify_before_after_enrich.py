#!/usr/bin/env python3
"""Verify before/after enrichment claim candidates with ms-swift."""

import argparse
import logging
import os
from typing import Any, Dict, List, Sequence

from utils import (
    DEFAULT_PROMPTS_YAML,
    build_swift_infer,
    configure_logging,
    extract_response_text,
    load_prompts,
    make_request,
    make_request_config,
    parse_json_response,
    read_jsonl,
    render_prompt,
    resolve_output_path,
    write_jsonl,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DEFAULT_INPUT_JSONL = os.path.join(
    PROJECT_DIR,
    "datasets",
    "claim_enrich",
    "target_enrich",
    "fr",
    "gemma4",
    "before_after_enrich_similarity.jsonl",
)
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "enrich_exp", "before_after_enrich_verify")
DEFAULT_OUTPUT_JSONL = "before_after_enrich_verify.jsonl"
DEFAULT_COMPACT_OUTPUT_JSONL = "before_after_enrich_verify_compact.jsonl"

DIRECTION_BEFORE_TO_AFTER = "before_to_after"
DIRECTION_AFTER_TO_BEFORE = "after_to_before"
DEFAULT_DIRECTIONS = [DIRECTION_BEFORE_TO_AFTER, DIRECTION_AFTER_TO_BEFORE]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify before/after enrichment similarity candidates. Model calls are deduplicated by "
            "canonical before->after claim pair. The after_to_before output reuses the canonical result "
            "with A/B and Enrichment direction inverted."
        )
    )
    parser.add_argument("--input-jsonl", type=str, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--prompts-yaml", type=str, default=DEFAULT_PROMPTS_YAML)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-jsonl", type=str, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--compact-output-jsonl", type=str, default=DEFAULT_COMPACT_OUTPUT_JSONL)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for debugging.")

    parser.add_argument("--model", type=str, required=True, help="Model id/path for ms-swift, e.g. Qwen/Qwen3.5-9B.")
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--model-revision", type=str, default=None)
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--infer-backend", type=str, default="vllm", choices=["vllm", "transformers", "sglang", "lmdeploy"])
    parser.add_argument("--template", type=str, default=None)
    parser.add_argument("--torch-dtype", type=str, default=None, choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-max-model-len", type=int, default=16384)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--add-non-thinking-prefix", action="store_true")

    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-file", type=str, default="run_before_after_enrich_verify.log")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def get_input_idx(sample: Dict[str, Any]) -> int:
    if "input_idx" not in sample:
        raise ValueError(f"Input sample is missing input_idx: qid={sample.get('qid')}")
    return int(sample["input_idx"])


def pair_key(direction: str, input_idx: int, qid: Any, a_idx: int, b_idx: int) -> str:
    return f"{direction}::{input_idx}::{qid}::{a_idx}::{b_idx}"


def canonical_pair_key(input_idx: int, qid: Any, before_idx: int, after_idx: int) -> str:
    return pair_key(DIRECTION_BEFORE_TO_AFTER, input_idx, qid, before_idx, after_idx)


def sample_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "input_idx": sample.get("input_idx"),
        "qid": sample.get("qid"),
        "name": sample.get("name"),
        "en_title": sample.get("en_title"),
        "target_title": sample.get("target_title"),
        "country_group": sample.get("country_group"),
        "target_lang": sample.get("target_lang"),
        "before_field": sample.get("before_field"),
        "after_field": sample.get("after_field"),
        "before_en_claim_count": sample.get("before_en_claim_count"),
        "before_target_claim_count": sample.get("before_target_claim_count"),
        "before_claim_count": sample.get("before_claim_count"),
        "after_claim_count": sample.get("after_claim_count"),
    }


def add_before_to_after_specs(
    sample: Dict[str, Any],
    input_idx: int,
    pair_specs: Dict[str, Dict[str, Any]],
) -> None:
    qid = sample.get("qid")
    meta = sample_meta(sample)
    for before_item in sample.get("before_claims", []) or []:
        before_idx = int(before_item["before_idx"])
        before_claim = before_item.get("before_claim", "")
        for candidate in before_item.get("after_matches", []) or []:
            after_idx = int(candidate["after_idx"])
            key = canonical_pair_key(input_idx, qid, before_idx, after_idx)
            spec = pair_specs.setdefault(key, {
                "pair_key": key,
                "direction": DIRECTION_BEFORE_TO_AFTER,
                "input_idx": input_idx,
                "qid": qid,
                "meta": meta,
                "a_label": "before",
                "b_label": "after",
                "a_idx": before_idx,
                "a_claim": before_claim,
                "b_idx": after_idx,
                "b_claim": candidate.get("after_claim", ""),
                "similarity": candidate.get("after_similarity"),
                "source_directions": [],
            })
            if DIRECTION_BEFORE_TO_AFTER not in spec["source_directions"]:
                spec["source_directions"].append(DIRECTION_BEFORE_TO_AFTER)


def add_after_to_before_specs(
    sample: Dict[str, Any],
    input_idx: int,
    pair_specs: Dict[str, Dict[str, Any]],
) -> None:
    qid = sample.get("qid")
    meta = sample_meta(sample)
    for after_item in sample.get("after_claims", []) or []:
        after_idx = int(after_item["after_idx"])
        after_claim = after_item.get("after_claim", "")
        for candidate in after_item.get("before_matches", []) or []:
            before_idx = int(candidate["before_idx"])
            key = canonical_pair_key(input_idx, qid, before_idx, after_idx)
            spec = pair_specs.setdefault(key, {
                "pair_key": key,
                "direction": DIRECTION_BEFORE_TO_AFTER,
                "input_idx": input_idx,
                "qid": qid,
                "meta": meta,
                "a_label": "before",
                "b_label": "after",
                "a_idx": before_idx,
                "a_claim": candidate.get("before_claim", ""),
                "b_idx": after_idx,
                "b_claim": after_claim,
                "similarity": candidate.get("before_similarity"),
                "source_directions": [],
            })
            if DIRECTION_AFTER_TO_BEFORE not in spec["source_directions"]:
                spec["source_directions"].append(DIRECTION_AFTER_TO_BEFORE)


def build_pair_specs(samples: Sequence[Dict[str, Any]], directions: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    pair_specs: Dict[str, Dict[str, Any]] = {}
    direction_set = set(directions)
    for sample in samples:
        input_idx = get_input_idx(sample)
        if DIRECTION_BEFORE_TO_AFTER in direction_set:
            add_before_to_after_specs(sample, input_idx, pair_specs)
        if DIRECTION_AFTER_TO_BEFORE in direction_set:
            add_after_to_before_specs(sample, input_idx, pair_specs)
    return pair_specs


def count_directional_candidate_pairs(samples: Sequence[Dict[str, Any]], directions: Sequence[str]) -> int:
    direction_set = set(directions)
    count = 0
    for sample in samples:
        if DIRECTION_BEFORE_TO_AFTER in direction_set:
            count += sum(len(item.get("after_matches", []) or []) for item in sample.get("before_claims", []) or [])
        if DIRECTION_AFTER_TO_BEFORE in direction_set:
            count += sum(len(item.get("before_matches", []) or []) for item in sample.get("after_claims", []) or [])
    return count


def build_prompt_values(pair_spec: Dict[str, Any]) -> Dict[str, Any]:
    meta = pair_spec.get("meta", {})
    return {
        "name": meta.get("name", ""),
        "en_title": meta.get("en_title", ""),
        "target_title": meta.get("target_title", ""),
        "a_claim": pair_spec["a_claim"],
        "b_claim": pair_spec["b_claim"],
    }


def infer_pair_specs(
    pair_specs: List[Dict[str, Any]],
    infer_pipeline: Any,
    request_config: Any,
    system_prompt: str,
    user_template: str,
    args: argparse.Namespace,
) -> Dict[str, Dict[str, Any]]:
    if not pair_specs:
        return {}
    if args.infer_backend == "vllm":
        chunks = [(0, pair_specs)]
    else:
        batch_size = 4
        chunks = [
            (start, pair_specs[start:start + batch_size])
            for start in range(0, len(pair_specs), batch_size)
        ]

    outputs: Dict[str, Dict[str, Any]] = {}
    for start, chunk in chunks:
        prompts = [render_prompt(user_template, build_prompt_values(pair_spec)) for pair_spec in chunk]
        requests = [make_request(system_prompt, prompt) for prompt in prompts]
        try:
            responses = infer_pipeline.infer(requests, request_config, use_tqdm=True)
            response_texts = [extract_response_text(response) for response in responses]
        except Exception as batch_error:
            logging.exception("Batch inference failed at pair_start=%d; falling back to single-item inference.", start)
            response_texts = []
            for request in requests:
                try:
                    response = infer_pipeline.infer([request], request_config, use_tqdm=False)[0]
                    response_texts.append(extract_response_text(response))
                except Exception as single_error:
                    response_texts.append(f"__ERROR__: {repr(single_error)} | batch_error: {repr(batch_error)}")

        for pair_spec, raw_response in zip(chunk, response_texts):
            parsed_response, parse_error = parse_json_response(raw_response)
            verification = {
                "pair_key": pair_spec["pair_key"],
                "direction": pair_spec["direction"],
                "a_label": pair_spec["a_label"],
                "b_label": pair_spec["b_label"],
                "a_idx": pair_spec["a_idx"],
                "a_claim": pair_spec["a_claim"],
                "b_idx": pair_spec["b_idx"],
                "b_claim": pair_spec["b_claim"],
                "similarity": pair_spec["similarity"],
                "parsed_response": parsed_response,
                "parse_error": parse_error,
                "raw_model_response": raw_response,
            }
            outputs[pair_spec["pair_key"]] = verification
        logging.info(
            "Verified %d/%d before/after candidate pairs.",
            min(start + len(chunk), len(pair_specs)),
            len(pair_specs),
        )
    return outputs


def empty_relation_index() -> Dict[str, List[int]]:
    return {
        "A=B": [],
        "A>B": [],
        "B>A": [],
        "A<>B": [],
        "contradicted": [],
        "not_relevant": [],
    }


def classify_relation(verification: Dict[str, Any]) -> str:
    parsed = verification.get("parsed_response")
    if not isinstance(parsed, dict):
        return "not_relevant"
    alignment = str(parsed.get("Alignment", "")).strip()
    enrichment = str(parsed.get("Enrichment", "")).strip()
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


def compact_response(parsed_response: Any) -> Dict[str, Any]:
    if not isinstance(parsed_response, dict):
        return {"Alignment": "", "Enrichment": ""}
    return {
        "Alignment": parsed_response.get("Alignment", ""),
        "Enrichment": parsed_response.get("Enrichment", ""),
    }


def invert_enrichment(enrichment: Any) -> Any:
    if not isinstance(enrichment, str):
        return enrichment
    if enrichment == "A > B":
        return "B > A"
    if enrichment == "B > A":
        return "A > B"
    return enrichment


def invert_parsed_response(parsed_response: Any) -> Any:
    if not isinstance(parsed_response, dict):
        return parsed_response
    inverted = dict(parsed_response)
    if str(inverted.get("Alignment", "")).strip() == "Aligned":
        inverted["Enrichment"] = invert_enrichment(inverted.get("Enrichment", ""))
    original_why = inverted.get("Why")
    if isinstance(original_why, str) and original_why:
        inverted["Why"] = (
            "Reused from canonical before_to_after verification with Claim A and Claim B inverted. "
            f"Original before_to_after explanation: {original_why}"
        )
    return inverted


def before_to_after_view(
    verification: Dict[str, Any],
    before_idx: int,
    before_claim: str,
    after_idx: int,
    after_claim: str,
    similarity: Any,
) -> Dict[str, Any]:
    view = dict(verification)
    view.update(
        {
            "pair_key": verification["pair_key"],
            "canonical_pair_key": verification["pair_key"],
            "direction": DIRECTION_BEFORE_TO_AFTER,
            "a_label": "before",
            "b_label": "after",
            "a_idx": before_idx,
            "a_claim": before_claim,
            "b_idx": after_idx,
            "b_claim": after_claim,
            "similarity": similarity,
            "reused_from_canonical": False,
        }
    )
    return view


def after_to_before_view(
    verification: Dict[str, Any],
    input_idx: int,
    qid: Any,
    after_idx: int,
    after_claim: str,
    before_idx: int,
    before_claim: str,
    similarity: Any,
) -> Dict[str, Any]:
    view = dict(verification)
    view.update(
        {
            "pair_key": pair_key(DIRECTION_AFTER_TO_BEFORE, input_idx, qid, after_idx, before_idx),
            "canonical_pair_key": verification["pair_key"],
            "direction": DIRECTION_AFTER_TO_BEFORE,
            "a_label": "after",
            "b_label": "before",
            "a_idx": after_idx,
            "a_claim": after_claim,
            "b_idx": before_idx,
            "b_claim": before_claim,
            "similarity": similarity,
            "parsed_response": invert_parsed_response(verification.get("parsed_response")),
            "reused_from_canonical": True,
        }
    )
    return view


def build_before_to_after_output(
    sample: Dict[str, Any],
    input_idx: int,
    pair_cache: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    qid = sample.get("qid")
    outputs = []
    for before_item in sample.get("before_claims", []) or []:
        before_idx = int(before_item["before_idx"])
        candidates = []
        relation_after_claim_idxs = empty_relation_index()
        for candidate in before_item.get("after_matches", []) or []:
            after_idx = int(candidate["after_idx"])
            after_claim = candidate.get("after_claim", "")
            key = canonical_pair_key(input_idx, qid, before_idx, after_idx)
            canonical_verification = pair_cache.get(key)
            verification = (
                before_to_after_view(
                    canonical_verification,
                    before_idx,
                    before_item.get("before_claim", ""),
                    after_idx,
                    after_claim,
                    candidate.get("after_similarity"),
                )
                if canonical_verification is not None
                else None
            )
            if verification is None:
                continue
            candidates.append(verification)
            relation_after_claim_idxs[classify_relation(verification)].append(after_idx)
        outputs.append(
            {
                "before_idx": before_idx,
                "before_claim": before_item.get("before_claim", ""),
                "relation_after_claim_idxs": relation_after_claim_idxs,
                "candidates": candidates,
            }
        )
    return outputs


def build_after_to_before_output(
    sample: Dict[str, Any],
    input_idx: int,
    pair_cache: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    qid = sample.get("qid")
    outputs = []
    for after_item in sample.get("after_claims", []) or []:
        after_idx = int(after_item["after_idx"])
        candidates = []
        relation_before_claim_idxs = empty_relation_index()
        for candidate in after_item.get("before_matches", []) or []:
            before_idx = int(candidate["before_idx"])
            before_claim = candidate.get("before_claim", "")
            key = canonical_pair_key(input_idx, qid, before_idx, after_idx)
            canonical_verification = pair_cache.get(key)
            verification = (
                after_to_before_view(
                    canonical_verification,
                    input_idx,
                    qid,
                    after_idx,
                    after_item.get("after_claim", ""),
                    before_idx,
                    before_claim,
                    candidate.get("before_similarity"),
                )
                if canonical_verification is not None
                else None
            )
            if verification is None:
                continue
            candidates.append(verification)
            relation_before_claim_idxs[classify_relation(verification)].append(before_idx)
        outputs.append(
            {
                "after_idx": after_idx,
                "after_claim": after_item.get("after_claim", ""),
                "relation_before_claim_idxs": relation_before_claim_idxs,
                "candidates": candidates,
            }
        )
    return outputs


def build_sample_output(
    sample: Dict[str, Any],
    pair_cache: Dict[str, Dict[str, Any]],
    directions: Sequence[str],
) -> Dict[str, Any]:
    input_idx = get_input_idx(sample)
    output = sample_meta(sample)
    direction_set = set(directions)
    if DIRECTION_BEFORE_TO_AFTER in direction_set:
        output["before_to_after_verifications"] = build_before_to_after_output(sample, input_idx, pair_cache)
    if DIRECTION_AFTER_TO_BEFORE in direction_set:
        output["after_to_before_verifications"] = build_after_to_before_output(sample, input_idx, pair_cache)
    return output


def compact_candidate(candidate: Dict[str, Any], output_claim_key: str) -> Dict[str, Any]:
    if candidate.get("direction") == DIRECTION_BEFORE_TO_AFTER:
        output_claim_idx_key = "after_idx"
        output_claim_idx = candidate.get("b_idx")
    else:
        output_claim_idx_key = "before_idx"
        output_claim_idx = candidate.get("b_idx")
    return {
        output_claim_idx_key: output_claim_idx,
        output_claim_key: candidate.get("b_claim"),
        "similarity": candidate.get("similarity"),
        "relation": classify_relation(candidate),
        "parsed_response": compact_response(candidate.get("parsed_response")),
    }


def build_compact_sample_output(full_output: Dict[str, Any]) -> Dict[str, Any]:
    compact = {
        "input_idx": full_output.get("input_idx"),
        "qid": full_output.get("qid"),
        "meta": {
            "name": full_output.get("name"),
            "country_group": full_output.get("country_group"),
            "target_lang": full_output.get("target_lang"),
        },
        "counts": {
            "before_en_claim_count": full_output.get("before_en_claim_count"),
            "before_target_claim_count": full_output.get("before_target_claim_count"),
            "before_claim_count": full_output.get("before_claim_count"),
            "after_claim_count": full_output.get("after_claim_count"),
        },
    }
    if "before_to_after_verifications" in full_output:
        compact["before_to_after_verifications"] = [
            {
                "before_idx": item.get("before_idx"),
                "before_claim": item.get("before_claim"),
                "relation_after_claim_idxs": item.get("relation_after_claim_idxs", empty_relation_index()),
                "candidates": [
                    compact_candidate(candidate, "after_claim")
                    for candidate in item.get("candidates", []) or []
                ],
            }
            for item in full_output.get("before_to_after_verifications", []) or []
        ]
    if "after_to_before_verifications" in full_output:
        compact["after_to_before_verifications"] = [
            {
                "after_idx": item.get("after_idx"),
                "after_claim": item.get("after_claim"),
                "relation_before_claim_idxs": item.get("relation_before_claim_idxs", empty_relation_index()),
                "candidates": [
                    compact_candidate(candidate, "before_claim")
                    for candidate in item.get("candidates", []) or []
                ],
            }
            for item in full_output.get("after_to_before_verifications", []) or []
        ]
    return compact


def main() -> None:
    args = parse_args()
    directions = DEFAULT_DIRECTIONS
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = resolve_output_path(results_dir, args.log_file) if str(args.log_file).strip() else ""
    configure_logging(args.log_level, log_file_path)

    samples = read_jsonl(args.input_jsonl, args.limit)
    system_prompt, user_template = load_prompts(args.prompts_yaml)
    pair_specs_by_key = build_pair_specs(samples, directions)
    pair_specs = list(pair_specs_by_key.values())
    directional_candidate_pair_count = count_directional_candidate_pairs(samples, directions)
    logging.info(
        "Loaded samples=%d | directions=%s | directional_candidate_pairs=%d | unique_canonical_pairs=%d | saved_model_calls=%d",
        len(samples),
        directions,
        directional_candidate_pair_count,
        len(pair_specs),
        directional_candidate_pair_count - len(pair_specs),
    )
    logging.info("Input JSONL: %s", args.input_jsonl)
    logging.info("Prompt YAML: %s", args.prompts_yaml)

    infer_pipeline = build_swift_infer(args)
    request_config = make_request_config(args)
    pair_cache = infer_pair_specs(
        pair_specs,
        infer_pipeline,
        request_config,
        system_prompt,
        user_template,
        args,
    )

    full_outputs = [
        build_sample_output(sample, pair_cache, directions)
        for sample in samples
    ]
    compact_outputs = [build_compact_sample_output(output) for output in full_outputs]

    output_jsonl = resolve_output_path(results_dir, args.output_jsonl)
    compact_output_jsonl = resolve_output_path(results_dir, args.compact_output_jsonl)
    write_jsonl(output_jsonl, full_outputs)
    write_jsonl(compact_output_jsonl, compact_outputs)
    logging.info("Saved details: %s", output_jsonl)
    logging.info("Saved compact details: %s", compact_output_jsonl)


if __name__ == "__main__":
    main()
