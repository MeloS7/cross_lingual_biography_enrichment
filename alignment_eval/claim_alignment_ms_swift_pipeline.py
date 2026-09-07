#!/usr/bin/env python3
"""Claim alignment pipeline with structured Alignment/Enrichment output via ms-swift."""

import argparse
import json
import logging
import os
from typing import Any, Dict, List

from swift.arguments import InferArguments
from swift.infer_engine import InferRequest, RequestConfig
from swift.pipelines.infer import SwiftInfer

from eval_utils import (
    DEFAULT_PROMPTS_YAML,
    DEFAULT_RESULTS_DIR,
    REPO_DIR,
    build_user_prompt,
    compute_metrics,
    extract_final_answer,
    extract_structured_prediction,
    get_claim_pair,
    get_gold_label,
    get_row_value,
    load_jsonl,
    load_prompts,
    resolve_results_path,
    write_jsonl,
)


SYSTEM_PROMPT, USER_PROMPT_TEMPLATE = load_prompts(DEFAULT_PROMPTS_YAML)
DEFAULT_INPUT_PATH = os.path.join(REPO_DIR, "datasets", "CLAW_4L_RC.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict relation labels for claim pairs via ms-swift.")
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Input JSONL where each row must contain reviewed_claim_a/reviewed_claim_b and label_type.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="claim_alignment_predictions.jsonl",
        help="Output prediction JSONL file name/path. Final file is always written under --results-dir.",
    )
    parser.add_argument(
        "--metrics-path",
        type=str,
        default="claim_alignment_metrics.json",
        help="Evaluation metrics JSON file name/path. Final file is always written under --results-dir.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory for all evaluation outputs.",
    )
    parser.add_argument(
        "--prompts-yaml",
        type=str,
        default=DEFAULT_PROMPTS_YAML,
        help="YAML file containing 'SYSTEM PROMPT' and 'USER PROMPT'.",
    )
    parser.add_argument("--model", type=str, required=True, help="Model id/path for ms-swift, e.g. Qwen/Qwen2.5-7B-Instruct.")
    parser.add_argument("--model-type", type=str, default=None, help="Optional model type for ms-swift.")
    parser.add_argument("--model-revision", type=str, default=None, help="Optional model revision.")
    parser.add_argument("--use-hf", action="store_true", help="Download/load model from HuggingFace Hub.")
    parser.add_argument(
        "--infer-backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm", "sglang", "lmdeploy"],
        help="Inference backend used by ms-swift.",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default=None,
        choices=["bfloat16", "float16", "float32"],
        help="Optional torch dtype. If omitted, ms-swift decides automatically.",
    )
    parser.add_argument(
        "--vllm-gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional vLLM gpu memory utilization (0~1).",
    )
    parser.add_argument(
        "--vllm-max-model-len",
        type=int,
        default=None,
        help="Optional vLLM max model length to reduce KV cache usage.",
    )
    parser.add_argument("--template", type=str, default=None, help="Optional template name for ms-swift.")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable model thinking mode when supported by the template/model.",
    )
    parser.add_argument(
        "--add-non-thinking-prefix",
        action="store_true",
        help="Add non-thinking prefix for models/templates that support dual modes.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=None,
        help="Optional max batch size override. Default is 8 for non-vllm backends; for vllm, rely on dynamic batching.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Max generated tokens per sample.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples for debugging.")
    parser.add_argument("--save-prompts", action="store_true", help="If set, save the full prompt in output JSONL.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for backend that supports it.")
    parser.add_argument(
        "--log-file",
        type=str,
        default="run.log",
        help="Run log file name/path. Final file is always written under --results-dir. Use an empty value to disable file logging.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def configure_logging(log_level: str, log_file_path: str = "") -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file_path:
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        handlers.append(logging.FileHandler(log_file_path, mode="w", encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def log_run_config(args: argparse.Namespace, results_dir: str, log_file_path: str) -> None:
    logging.info(
        "Run config | model=%s | model_type=%s | infer_backend=%s | torch_dtype=%s | temperature=%s | top_p=%s | top_k=%s | max_new_tokens=%s | max_batch_size=%s | vllm_max_model_len=%s | vllm_gpu_memory_utilization=%s | prompts_yaml=%s | input_path=%s | results_dir=%s | log_file=%s",
        args.model,
        args.model_type,
        args.infer_backend,
        args.torch_dtype,
        args.temperature,
        args.top_p,
        args.top_k,
        args.max_new_tokens,
        args.max_batch_size,
        args.vllm_max_model_len,
        args.vllm_gpu_memory_utilization,
        args.prompts_yaml,
        args.input_path,
        results_dir,
        log_file_path or "",
    )


def build_swift_infer(args: argparse.Namespace) -> SwiftInfer:
    if args.max_batch_size is not None:
        effective_max_batch_size = max(1, args.max_batch_size)
    elif args.infer_backend == "vllm":
        effective_max_batch_size = None
    else:
        effective_max_batch_size = 8

    infer_kwargs: Dict[str, Any] = dict(
        model=args.model,
        model_type=args.model_type,
        model_revision=args.model_revision,
        use_hf=args.use_hf,
        task_type="causal_lm",
        infer_backend=args.infer_backend,
        torch_dtype=args.torch_dtype,
        template=args.template,
        enable_thinking=args.enable_thinking,
        add_non_thinking_prefix=args.add_non_thinking_prefix,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
    )
    if effective_max_batch_size is not None:
        infer_kwargs["max_batch_size"] = effective_max_batch_size
    if args.vllm_gpu_memory_utilization is not None:
        infer_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.vllm_max_model_len is not None:
        infer_kwargs["vllm_max_model_len"] = args.vllm_max_model_len
    infer_args = InferArguments(**infer_kwargs)
    logging.info(
        "ms-swift resolved config | model=%s | model_type=%s | template=%s | model_dir=%s | torch_dtype=%s",
        infer_args.model,
        infer_args.model_type,
        infer_args.template,
        getattr(infer_args, "model_dir", None),
        infer_args.torch_dtype,
    )
    return SwiftInfer(infer_args)


def extract_response_text(response: Any) -> str:
    # ChatCompletionResponse style.
    try:
        return response.choices[0].message.content
    except Exception:
        return str(response)


def predict_all(rows: List[Dict[str, Any]], infer_pipeline: SwiftInfer, args: argparse.Namespace) -> List[Dict[str, Any]]:
    request_config = RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        stream=False,
    )

    outputs: List[Dict[str, Any]] = []
    if args.max_batch_size is not None:
        bs = max(1, args.max_batch_size)
    elif args.infer_backend == "vllm":
        # Let vLLM handle dynamic batching instead of fixed-size client chunking.
        bs = len(rows)
    else:
        bs = 8
    # Always use backend/ms-swift native tqdm progress across all backends.
    infer_use_tqdm = True
    batch_starts = range(0, len(rows), bs)

    for start in batch_starts:
        chunk = rows[start:start + bs]
        requests: List[InferRequest] = []
        prompts: List[str] = []

        for row in chunk:
            claim_a, claim_b = get_claim_pair(row)
            user_prompt = build_user_prompt(row, claim_a, claim_b, USER_PROMPT_TEMPLATE)
            prompts.append(user_prompt)
            requests.append(
                InferRequest(messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ]))

        try:
            responses = infer_pipeline.infer(requests, request_config, use_tqdm=infer_use_tqdm)
            texts = [extract_response_text(resp) for resp in responses]
        except Exception as e:
            logging.exception("Batch inference failed at start=%d. Falling back to single-item inference.", start)
            texts = []
            for req in requests:
                try:
                    single_resp = infer_pipeline.infer([req], request_config, use_tqdm=False)[0]
                    texts.append(extract_response_text(single_resp))
                except Exception as single_e:
                    texts.append(f"__ERROR__: {repr(single_e)} | batch_error: {repr(e)}")

        for row, prompt, raw_pred in zip(chunk, prompts, texts):
            final_answer = extract_final_answer(raw_pred)
            pred_alignment, pred_enrichment, pred_label, pred_why = extract_structured_prediction(raw_pred)
            gold_label = get_gold_label(row)
            claim_a, claim_b = get_claim_pair(row)
            out = {
                "qid": get_row_value(row, "qid"),
                "name": get_row_value(row, "name"),
                "country_group": get_row_value(row, "country_group"),
                "gold_label": gold_label,
                "pred_label": pred_label,
                "pred_alignment": pred_alignment,
                "pred_enrichment": pred_enrichment,
                "pred_why": pred_why,
                "pred_json": {
                    "Why": pred_why,
                    "Alignment": pred_alignment,
                    "Enrichment": pred_enrichment,
                },
                "raw_prediction": final_answer,
                "raw_model_response": raw_pred,
                "final_answer_segment": final_answer,
                "claim_a": claim_a,
                "claim_b": claim_b,
            }
            if args.save_prompts:
                out["prompt"] = prompt
            outputs.append(out)

        logging.info("Processed %d/%d", min(start + bs, len(rows)), len(rows))
    return outputs


def main() -> None:
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    log_file_path = resolve_results_path(results_dir, args.log_file) if str(args.log_file).strip() else ""
    configure_logging(args.log_level, log_file_path)

    global SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE = load_prompts(args.prompts_yaml)

    rows = load_jsonl(args.input_path, args.limit)
    if not rows:
        raise ValueError(f"No data loaded from {args.input_path}")

    output_jsonl_path = resolve_results_path(results_dir, args.output_jsonl)
    metrics_path = resolve_results_path(results_dir, args.metrics_path)

    logging.info("Loaded %d samples from %s", len(rows), args.input_path)
    logging.info("All outputs will be saved under: %s", results_dir)
    if log_file_path:
        logging.info("Run log will be saved to: %s", log_file_path)
    log_run_config(args, results_dir, log_file_path)
    infer_pipeline = build_swift_infer(args)
    pred_rows = predict_all(rows, infer_pipeline, args)

    metrics = compute_metrics(pred_rows)
    write_jsonl(output_jsonl_path, pred_rows)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logging.info("Saved predictions to %s", output_jsonl_path)
    logging.info("Saved metrics to %s", metrics_path)
    logging.info("Accuracy=%.4f | F1=%.4f", metrics["accuracy"], metrics["f1"])


if __name__ == "__main__":
    main()
