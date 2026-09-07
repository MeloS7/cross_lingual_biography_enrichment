#!/usr/bin/env python3
"""Verify retrieved target-claim to English-claim candidates with ms-swift."""

import argparse
import logging
import os
from typing import Any, Dict, List, Sequence, Tuple

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
    "results",
    "enrich_exp",
    "enrich_claim_similarity",
    "target_to_en_claim_top5_mpnet.jsonl",
)
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "enrich_exp", "target_en_enrichment")
DEFAULT_OUTPUT_JSONL = "target_en_claim_candidate_verify.jsonl"
DEFAULT_COMPACT_OUTPUT_JSONL = "target_en_claim_candidate_verify_compact.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify retrieved target-to-English claim candidates. Claim A is the target-language biography "
            "claim and Claim B is the English biography claim, so A > B means target has extra information."
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
    parser.add_argument("--log-file", type=str, default="run_target_en_claim_verify.log")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def pair_key(sample_idx: int, qid: Any, target_claim_idx: int, en_claim_idx: int) -> str:
    return f"{sample_idx}::{qid}::{target_claim_idx}::{en_claim_idx}"


def build_pair_specs(samples: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    pair_specs: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        sample_idx = int(sample.get("sample_idx", len(pair_specs)))
        qid = sample.get("qid")
        meta = sample.get("meta", {})
        for target_item in sample.get("target_to_en_candidates", []) or []:
            target_claim_idx = int(target_item["target_claim_idx"])
            target_claim = target_item.get("target_claim", "")
            for candidate in target_item.get("candidates", []) or []:
                en_claim_idx = int(candidate["en_claim_idx"])
                key = pair_key(sample_idx, qid, target_claim_idx, en_claim_idx)
                pair_specs[key] = {
                    "pair_key": key,
                    "sample_idx": sample_idx,
                    "qid": qid,
                    "meta": meta,
                    "target_claim_idx": target_claim_idx,
                    "target_claim": target_claim,
                    "en_claim_idx": en_claim_idx,
                    "en_claim": candidate.get("en_claim", ""),
                    "similarity": candidate.get("similarity"),
                }
    return pair_specs


def build_prompt_values(pair_spec: Dict[str, Any]) -> Dict[str, Any]:
    meta = pair_spec.get("meta", {})
    return {
        "name": meta.get("name", ""),
        "en_title": meta.get("en_title", ""),
        "target_title": meta.get("target_title", ""),
        "a_claim": pair_spec["target_claim"],
        "b_claim": pair_spec["en_claim"],
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
                "target_claim_idx": pair_spec["target_claim_idx"],
                "target_claim": pair_spec["target_claim"],
                "en_claim_idx": pair_spec["en_claim_idx"],
                "en_claim": pair_spec["en_claim"],
                "similarity": pair_spec["similarity"],
                "parsed_response": parsed_response,
                "parse_error": parse_error,
                "raw_model_response": raw_response,
            }
            outputs[pair_spec["pair_key"]] = verification
        logging.info(
            "Verified %d/%d candidate pairs.",
            min(start + len(chunk), len(pair_specs)),
            len(pair_specs),
        )
    return outputs


def build_sample_output(sample: Dict[str, Any], pair_cache: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    sample_idx = int(sample.get("sample_idx", 0))
    qid = sample.get("qid")
    verified_targets = []
    for target_item in sample.get("target_to_en_candidates", []) or []:
        target_claim_idx = int(target_item["target_claim_idx"])
        candidates = []
        relation_en_claim_idxs = empty_relation_index()
        for candidate in target_item.get("candidates", []) or []:
            en_claim_idx = int(candidate["en_claim_idx"])
            key = pair_key(sample_idx, qid, target_claim_idx, en_claim_idx)
            verification = pair_cache.get(key)
            if verification is None:
                continue
            candidates.append(verification)
            relation_en_claim_idxs[classify_relation(verification)].append(en_claim_idx)
        verified_targets.append(
            {
                "target_claim_idx": target_claim_idx,
                "target_claim": target_item.get("target_claim", ""),
                "relation_en_claim_idxs": relation_en_claim_idxs,
                "candidates": candidates,
            }
        )
    return {
        "sample_idx": sample.get("sample_idx"),
        "qid": qid,
        "meta": sample.get("meta", {}),
        "en_claim_count": sample.get("en_claim_count"),
        "target_claim_count": sample.get("target_claim_count"),
        "target_to_en_verifications": verified_targets,
    }


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


def build_compact_sample_output(full_output: Dict[str, Any]) -> Dict[str, Any]:
    compact_targets = []
    for target_item in full_output.get("target_to_en_verifications", []) or []:
        compact_targets.append(
            {
                "target_claim_idx": target_item.get("target_claim_idx"),
                "target_claim": target_item.get("target_claim"),
                "relation_en_claim_idxs": target_item.get("relation_en_claim_idxs", empty_relation_index()),
                "candidates": [
                    {
                        "en_claim_idx": candidate.get("en_claim_idx"),
                        "en_claim": candidate.get("en_claim"),
                        "similarity": candidate.get("similarity"),
                        "relation": classify_relation(candidate),
                        "parsed_response": compact_response(candidate.get("parsed_response")),
                    }
                    for candidate in target_item.get("candidates", []) or []
                ],
            }
        )
    return {
        "sample_idx": full_output.get("sample_idx"),
        "qid": full_output.get("qid"),
        "meta": full_output.get("meta", {}),
        "en_claim_count": full_output.get("en_claim_count"),
        "target_claim_count": full_output.get("target_claim_count"),
        "target_to_en_verifications": compact_targets,
    }


def main() -> None:
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = resolve_output_path(results_dir, args.log_file) if str(args.log_file).strip() else ""
    configure_logging(args.log_level, log_file_path)

    samples = read_jsonl(args.input_jsonl, args.limit)
    system_prompt, user_template = load_prompts(args.prompts_yaml)
    pair_specs_by_key = build_pair_specs(samples)
    pair_specs = list(pair_specs_by_key.values())
    logging.info("Loaded samples=%d | candidate_pairs=%d", len(samples), len(pair_specs))
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

    full_outputs = [build_sample_output(sample, pair_cache) for sample in samples]
    compact_outputs = [build_compact_sample_output(output) for output in full_outputs]
    output_jsonl = resolve_output_path(results_dir, args.output_jsonl)
    compact_output_jsonl = resolve_output_path(results_dir, args.compact_output_jsonl)
    write_jsonl(output_jsonl, full_outputs)
    write_jsonl(compact_output_jsonl, compact_outputs)
    logging.info("Saved details: %s", output_jsonl)
    logging.info("Saved compact details: %s", compact_output_jsonl)


if __name__ == "__main__":
    main()
