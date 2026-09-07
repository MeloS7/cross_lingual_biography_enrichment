#!/usr/bin/env python3
"""Claim alignment pipeline with structured Alignment/Enrichment output via OpenRouter API."""

import argparse
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm.auto import tqdm

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
    parser = argparse.ArgumentParser(description="Predict relation labels for claim pairs via OpenRouter API.")
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
        "--cost-log-jsonl",
        type=str,
        default="claim_alignment_cost_openrouter.jsonl",
        help="Cost/usage log JSONL file name/path. Final file is always written under --results-dir.",
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
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-5.1",
        help="OpenRouter model id, e.g. openai/gpt-5.1.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenRouter API key. If omitted, read OPENROUTER_API_KEY from env.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="OpenRouter API base url.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16384, help="Max generated tokens per sample.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Decoding temperature.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling.")
    parser.add_argument("--top-k", type=int, default=None, help="Top-k sampling.")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout seconds.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries for transient errors.")
    parser.add_argument("--retry-wait", type=float, default=2.0, help="Base backoff seconds.")
    parser.add_argument("--http-referer", type=str, default=None, help="Optional HTTP-Referer header.")
    parser.add_argument("--x-title", type=str, default="claim-alignment-eval", help="Optional X-Title header.")
    parser.add_argument("--check-account-credits", action="store_true", help="Query /credits before and after run.")
    parser.add_argument("--log-every", type=int, default=50, help="Log cumulative cost every N samples.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples for debugging.")
    parser.add_argument("--save-prompts", action="store_true", help="If set, save the full prompt in output JSONL.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def extract_reasoning_usage(usage: Dict[str, Any], response: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    reasoning_tokens = to_int(
        usage.get("reasoning_tokens", usage.get("reasoning", {}).get("tokens") if isinstance(usage.get("reasoning"), dict) else 0),
        0,
    )
    completion_details = usage.get("completion_tokens_details")
    if isinstance(completion_details, dict):
        reasoning_tokens = to_int(completion_details.get("reasoning_tokens"), reasoning_tokens)

    response_reasoning = None
    if isinstance(response, dict):
        try:
            response_reasoning = response["choices"][0]["message"].get("reasoning")
        except Exception:
            response_reasoning = None

    return {
        "request_reasoning_config": None,
        "reasoning_effort": None,
        "reasoning_tokens": reasoning_tokens,
        "response_reasoning_present": response_reasoning is not None,
    }


def summarize_reasoning_usage(cost_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(cost_logs)
    buckets = {
        "no_reasoning_tokens": 0,
        "has_reasoning_tokens": 0,
        "unknown": 0,
    }
    token_values: List[int] = []
    response_reasoning_present = 0

    for row in cost_logs:
        reasoning_usage = row.get("reasoning_usage") if isinstance(row.get("reasoning_usage"), dict) else {}
        tokens = reasoning_usage.get("reasoning_tokens")
        if tokens is None:
            buckets["unknown"] += 1
        else:
            token_count = to_int(tokens, 0)
            token_values.append(token_count)
            if token_count > 0:
                buckets["has_reasoning_tokens"] += 1
            else:
                buckets["no_reasoning_tokens"] += 1
        if reasoning_usage.get("response_reasoning_present"):
            response_reasoning_present += 1

    bucket_percentages = {
        key: (value / total if total else 0.0)
        for key, value in buckets.items()
    }
    positive_values = [x for x in token_values if x > 0]
    return {
        "note": "The request does not set reasoning.effort. OpenRouter/OpenAI usually returns reasoning token counts, not a named default effort per sample.",
        "total_samples": total,
        "bucket_counts": buckets,
        "bucket_percentages": bucket_percentages,
        "response_reasoning_present_count": response_reasoning_present,
        "response_reasoning_present_percentage": response_reasoning_present / total if total else 0.0,
        "reasoning_tokens": {
            "total": sum(token_values),
            "mean_all_samples": sum(token_values) / len(token_values) if token_values else 0.0,
            "mean_positive_samples": sum(positive_values) / len(positive_values) if positive_values else 0.0,
            "max": max(token_values) if token_values else 0,
        },
    }


def http_json_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    payload: Optional[Dict[str, Any]] = None,
    timeout: int = 120,
) -> Dict[str, Any]:
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = Request(url=url, method=method, headers=headers, data=data)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        msg = f"HTTP {e.code} calling {url}: {body[:500]}"
        raise RuntimeError(msg) from e
    except URLError as e:
        raise RuntimeError(f"Network error calling {url}: {repr(e)}") from e


def fetch_account_credits(base_url: str, headers: Dict[str, str], timeout: int) -> Optional[Dict[str, float]]:
    credits_url = base_url.rstrip("/") + "/credits"
    try:
        data = http_json_request("GET", credits_url, headers, payload=None, timeout=timeout)
    except Exception as e:
        logging.warning("Failed to fetch /credits: %s", repr(e))
        return None

    payload = data.get("data", {}) if isinstance(data, dict) else {}
    total_credits = to_float(payload.get("total_credits"), 0.0)
    total_usage = to_float(payload.get("total_usage"), 0.0)
    return {
        "total_credits": total_credits,
        "total_usage": total_usage,
        "remaining_credits": total_credits - total_usage,
    }


def call_chat_completion(
    args: argparse.Namespace,
    headers: Dict[str, str],
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    payload: Dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "stream": False,
        "max_completion_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
    }
    if args.top_k is not None:
        payload["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        payload["repetition_penalty"] = args.repetition_penalty

    retryable_markers = ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504", "Network error")
    last_error: Optional[Exception] = None

    for attempt in range(args.max_retries + 1):
        try:
            return http_json_request("POST", url, headers, payload=payload, timeout=args.timeout)
        except Exception as e:  # pragma: no cover
            last_error = e
            message = str(e)
            retryable = any(marker in message for marker in retryable_markers)
            if (not retryable) or attempt >= args.max_retries:
                break
            sleep_s = args.retry_wait * (2 ** attempt)
            logging.warning(
                "OpenRouter request failed (attempt %d/%d), retry in %.1fs: %s",
                attempt + 1,
                args.max_retries + 1,
                sleep_s,
                message,
            )
            time.sleep(sleep_s)

    raise RuntimeError(f"OpenRouter request failed after retries: {repr(last_error)}")


def extract_response_text(response: Dict[str, Any]) -> str:
    try:
        return response["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(response, ensure_ascii=False)


def build_headers(args: argparse.Namespace, api_key: str) -> Dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if args.http_referer:
        headers["HTTP-Referer"] = args.http_referer
    if args.x_title:
        headers["X-Title"] = args.x_title
    return headers


def predict_all(
    rows: List[Dict[str, Any]],
    args: argparse.Namespace,
    headers: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    cost_logs: List[Dict[str, Any]] = []

    totals = {
        "requests": 0,
        "errors": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost_credits": 0.0,
        "missing_usage": 0,
    }

    indices = tqdm(range(len(rows)), total=len(rows), desc="Evaluating", unit="sample")

    for idx in indices:
        row = rows[idx]
        claim_a, claim_b = get_claim_pair(row)
        user_prompt = build_user_prompt(row, claim_a, claim_b, USER_PROMPT_TEMPLATE)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        raw_pred = ""
        response_id = None
        response_model = None
        response_data: Optional[Dict[str, Any]] = None
        usage: Dict[str, Any] = {}
        request_error = None

        try:
            response_data = call_chat_completion(args, headers, messages)
            raw_pred = extract_response_text(response_data)
            response_id = response_data.get("id")
            response_model = response_data.get("model")
            usage = response_data.get("usage", {}) if isinstance(response_data, dict) else {}
        except Exception as e:
            request_error = repr(e)
            raw_pred = f"__ERROR__: {request_error}"
            logging.error("Sample %d request failed: %s", idx, request_error)

        final_answer = extract_final_answer(raw_pred)
        pred_alignment, pred_enrichment, pred_label, pred_why = extract_structured_prediction(raw_pred)
        gold_label = get_gold_label(row)

        prompt_tokens = to_int(usage.get("prompt_tokens"), 0)
        completion_tokens = to_int(usage.get("completion_tokens"), 0)
        total_tokens = to_int(usage.get("total_tokens"), prompt_tokens + completion_tokens)
        cost_credits = to_float(usage.get("cost"), 0.0)
        reasoning_usage = extract_reasoning_usage(usage, response_data)

        totals["requests"] += 1
        totals["prompt_tokens"] += prompt_tokens
        totals["completion_tokens"] += completion_tokens
        totals["total_tokens"] += total_tokens
        totals["total_cost_credits"] += cost_credits

        if request_error is not None:
            totals["errors"] += 1
        if not usage:
            totals["missing_usage"] += 1

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
            "openrouter_request_id": response_id,
            "openrouter_model": response_model or args.model,
            "openrouter_reasoning_usage": reasoning_usage,
            "openrouter_usage": usage,
            "openrouter_cost_credits": cost_credits,
            "request_error": request_error,
        }
        if args.save_prompts:
            out["prompt"] = user_prompt
        outputs.append(out)

        cost_logs.append(
            {
                "index": idx,
                "qid": get_row_value(row, "qid"),
                "openrouter_request_id": response_id,
                "model": response_model or args.model,
                "reasoning_usage": reasoning_usage,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_credits": cost_credits,
                "request_error": request_error,
            }
        )

        if args.log_every > 0 and (idx + 1) % args.log_every == 0:
            logging.info(
                "Processed %d/%d | cumulative_cost_credits=%.8f | total_tokens=%d",
                idx + 1,
                len(rows),
                totals["total_cost_credits"],
                totals["total_tokens"],
            )

    usage_summary = {
        **totals,
        "request_reasoning_config": None,
        "avg_cost_credits_per_request": (totals["total_cost_credits"] / totals["requests"]) if totals["requests"] else 0.0,
        "avg_tokens_per_request": (totals["total_tokens"] / totals["requests"]) if totals["requests"] else 0.0,
    }
    return outputs, cost_logs, usage_summary


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    global SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    SYSTEM_PROMPT, USER_PROMPT_TEMPLATE = load_prompts(args.prompts_yaml)

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key not found. Pass --api-key or set OPENROUTER_API_KEY.")

    rows = load_jsonl(args.input_path, args.limit)
    if not rows:
        raise ValueError(f"No data loaded from {args.input_path}")

    results_dir = os.path.abspath(args.results_dir)
    os.makedirs(results_dir, exist_ok=True)
    output_jsonl_path = resolve_results_path(results_dir, args.output_jsonl)
    metrics_path = resolve_results_path(results_dir, args.metrics_path)
    cost_log_path = resolve_results_path(results_dir, args.cost_log_jsonl)

    headers = build_headers(args, api_key)

    logging.info("Loaded %d samples from %s", len(rows), args.input_path)
    logging.info("All outputs will be saved under: %s", results_dir)
    credits_before = fetch_account_credits(args.base_url, headers, args.timeout) if args.check_account_credits else None
    if credits_before is not None:
        logging.info(
            "Account before | total_usage=%.6f credits | remaining=%.6f credits",
            credits_before["total_usage"],
            credits_before["remaining_credits"],
        )

    pred_rows, cost_logs, usage_summary = predict_all(rows, args, headers)

    metrics = compute_metrics(pred_rows)
    metrics["openrouter_usage_summary"] = usage_summary
    metrics["openrouter_reasoning_summary"] = summarize_reasoning_usage(cost_logs)
    credits_after = fetch_account_credits(args.base_url, headers, args.timeout) if args.check_account_credits else None

    write_jsonl(output_jsonl_path, pred_rows)
    write_jsonl(cost_log_path, cost_logs)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    logging.info("Saved predictions to %s", output_jsonl_path)
    logging.info("Saved cost log to %s", cost_log_path)
    logging.info("Saved metrics to %s", metrics_path)
    logging.info("Accuracy=%.4f | F1=%.4f", metrics["accuracy"], metrics["f1"])
    logging.info(
        "Usage Summary | total_cost_credits=%.8f | prompt_tokens=%d | completion_tokens=%d | errors=%d",
        usage_summary["total_cost_credits"],
        usage_summary["prompt_tokens"],
        usage_summary["completion_tokens"],
        usage_summary["errors"],
    )
    reasoning_summary = metrics["openrouter_reasoning_summary"]
    logging.info(
        "Reasoning Summary | has_reasoning_tokens=%d/%d (%.2f%%) | no_reasoning_tokens=%d/%d (%.2f%%) | total_reasoning_tokens=%d",
        reasoning_summary["bucket_counts"]["has_reasoning_tokens"],
        reasoning_summary["total_samples"],
        reasoning_summary["bucket_percentages"]["has_reasoning_tokens"] * 100,
        reasoning_summary["bucket_counts"]["no_reasoning_tokens"],
        reasoning_summary["total_samples"],
        reasoning_summary["bucket_percentages"]["no_reasoning_tokens"] * 100,
        reasoning_summary["reasoning_tokens"]["total"],
    )
    if credits_before is not None and credits_after is not None:
        logging.info(
            "Account delta | usage=%.8f credits | remaining=%.8f credits",
            credits_after["total_usage"] - credits_before["total_usage"],
            credits_after["remaining_credits"] - credits_before["remaining_credits"],
        )


if __name__ == "__main__":
    main()
