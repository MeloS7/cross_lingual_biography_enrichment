#!/usr/bin/env python3
"""Evaluate claim-pair relations with DeBERTa NLI and MiniCheck."""

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EVAL_DIR)
DEFAULT_INPUT_PATH = os.path.join(REPO_DIR, "datasets", "CLAW_4L_RC.jsonl")
DEFAULT_RESULTS_DIR = os.path.join(REPO_DIR, "results", "alignment_eval", "claim_relation_nli_minicheck")
DEFAULT_DEBERTA_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
DEFAULT_MINICHECK_MODEL = "Bespoke-MiniCheck-7B"
DEFAULT_MINICHECK_CACHE_DIR = os.path.join(EVAL_DIR, "ckpts")
EVAL_LABELS = [
    "aligned",
    "contradicted",
    "not relevant",
    "partial aligned (A>B)",
    "partial aligned (B>A)",
    "partial aligned (A<>B)",
]
INVALID_LABEL = "__invalid__"
DEBERTA_LABEL_NAMES = ["entailment", "neutral", "contradiction"]
SUPPORTED_LABELS = {
    "deberta": {
        "aligned",
        "contradicted",
        "partial aligned (A>B)",
        "partial aligned (B>A)",
    },
    "minicheck": {
        "aligned",
        "partial aligned (A>B)",
        "partial aligned (B>A)",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CLAW_4L_RC claim pairs with bidirectional DeBERTa NLI "
            "and/or Bespoke-MiniCheck-7B entailment."
        )
    )
    parser.add_argument("--input-path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-jsonl", type=str, default="claim_relation_predictions.jsonl")
    parser.add_argument("--metrics-json", type=str, default="claim_relation_metrics.json")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deberta", "minicheck"],
        choices=["deberta", "minicheck"],
        help="Which evaluators to run.",
    )
    parser.add_argument("--deberta-model", type=str, default=DEFAULT_DEBERTA_MODEL)
    parser.add_argument("--minicheck-model", type=str, default=DEFAULT_MINICHECK_MODEL)
    parser.add_argument("--minicheck-cache-dir", type=str, default=DEFAULT_MINICHECK_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DeBERTa.")
    parser.add_argument(
        "--minicheck-chunk-size",
        type=int,
        default=None,
        help="Optional chunk_size passed to MiniCheck.score.",
    )
    parser.add_argument("--device", type=str, default=None, help="Device for DeBERTa, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_label(label: Any) -> str:
    text = safe_text(label).lower()
    aliases = {
        "aligned": "aligned",
        "contradict": "contradicted",
        "contradicted": "contradicted",
        "contradiction": "contradicted",
        "not relevant": "not relevant",
        "partial aligned (a>b)": "partial aligned (A>B)",
        "partial aligned (b>a)": "partial aligned (B>A)",
        "partial aligned (a<>b)": "partial aligned (A<>B)",
    }
    return aliases.get(text, INVALID_LABEL)


def get_claim_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    claim_a = row.get("reviewed_claim_a") if isinstance(row.get("reviewed_claim_a"), dict) else {}
    claim_b = row.get("reviewed_claim_b") if isinstance(row.get("reviewed_claim_b"), dict) else {}
    return safe_text(claim_a.get("claim")), safe_text(claim_b.get("claim"))


def base_row_output(row: Dict[str, Any], fallback_idx: int) -> Dict[str, Any]:
    meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
    claim_a, claim_b = get_claim_pair(row)
    return {
        "row_index": row.get("row_index", fallback_idx),
        "qid": row.get("qid", meta.get("qid", "")),
        "gold_label": normalize_label(row.get("label_type")),
        "raw_label_type": row.get("label_type"),
        "country_group": meta.get("country_group", ""),
        "target_lang": meta.get("target_lang", ""),
        "name": meta.get("name", ""),
        "en_title": meta.get("en_title", ""),
        "target_title": meta.get("target_title", ""),
        "claim_a": claim_a,
        "claim_b": claim_b,
        "predictions": {},
    }


def supported_row_indices(output_rows: Sequence[Dict[str, Any]], model_key: str) -> List[int]:
    supported_labels = SUPPORTED_LABELS[model_key]
    return [
        idx
        for idx, row in enumerate(output_rows)
        if row.get("gold_label") in supported_labels
    ]


def skipped_prediction(model_name: str, model_key: str, gold_label: str) -> Dict[str, Any]:
    return {
        "model": model_name,
        "predicted_label": None,
        "skipped": True,
        "skip_reason": f"gold_label={gold_label!r} is outside {model_key} supported labels",
        "supported_labels": sorted(SUPPORTED_LABELS[model_key]),
    }


def relation_from_deberta(ab_label: str, ba_label: str) -> str:
    if ab_label == "contradiction" or ba_label == "contradiction":
        return "contradicted"
    if ab_label == "entailment" and ba_label == "entailment":
        return "aligned"
    if ab_label == "entailment" and ba_label == "neutral":
        return "partial aligned (A>B)"
    if ab_label == "neutral" and ba_label == "entailment":
        return "partial aligned (B>A)"
    return INVALID_LABEL


def relation_from_minicheck(ab_label: int, ba_label: int) -> str:
    if ab_label == 1 and ba_label == 1:
        return "aligned"
    if ab_label == 1 and ba_label == 0:
        return "partial aligned (A>B)"
    if ab_label == 0 and ba_label == 1:
        return "partial aligned (B>A)"
    return INVALID_LABEL


def predict_deberta(rows: Sequence[Dict[str, Any]], model_name: str, batch_size: int, device_arg: Optional[str]) -> List[Dict[str, Any]]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device(device_arg) if device_arg else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    pairs: List[Tuple[str, str]] = []
    for row in rows:
        claim_a, claim_b = get_claim_pair(row)
        pairs.append((claim_a, claim_b))
        pairs.append((claim_b, claim_a))

    pair_outputs: List[Dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            premises = [premise for premise, _hypothesis in chunk]
            hypotheses = [hypothesis for _premise, hypothesis in chunk]
            inputs = tokenizer(premises, hypotheses, truncation=True, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
            for prob in probs:
                label_idx = max(range(len(prob)), key=lambda idx: prob[idx])
                pair_outputs.append(
                    {
                        "label": DEBERTA_LABEL_NAMES[label_idx],
                        "probabilities": {
                            label: float(prob[idx])
                            for idx, label in enumerate(DEBERTA_LABEL_NAMES)
                        },
                    }
                )
            print(f"DeBERTa scored {min(start + batch_size, len(pairs))}/{len(pairs)} directional pairs")

    predictions: List[Dict[str, Any]] = []
    for row_idx in range(len(rows)):
        ab = pair_outputs[row_idx * 2]
        ba = pair_outputs[row_idx * 2 + 1]
        predictions.append(
            {
                "model": model_name,
                "a_to_b": ab,
                "b_to_a": ba,
                "predicted_label": relation_from_deberta(ab["label"], ba["label"]),
            }
        )
    return predictions


def predict_minicheck(
    rows: Sequence[Dict[str, Any]],
    model_name: str,
    cache_dir: str,
    chunk_size: Optional[int],
) -> List[Dict[str, Any]]:
    from minicheck.minicheck import MiniCheck

    scorer = MiniCheck(model_name=model_name, enable_prefix_caching=False, cache_dir=cache_dir)
    docs: List[str] = []
    claims: List[str] = []
    for row in rows:
        claim_a, claim_b = get_claim_pair(row)
        docs.extend([claim_a, claim_b])
        claims.extend([claim_b, claim_a])

    score_kwargs: Dict[str, Any] = {}
    if chunk_size is not None:
        score_kwargs["chunk_size"] = chunk_size
    pred_labels, raw_probs, _, _ = scorer.score(docs=docs, claims=claims, **score_kwargs)

    predictions: List[Dict[str, Any]] = []
    for row_idx in range(len(rows)):
        ab_label = int(pred_labels[row_idx * 2])
        ba_label = int(pred_labels[row_idx * 2 + 1])
        predictions.append(
            {
                "model": model_name,
                "a_to_b": {
                    "label": ab_label,
                    "probability": float(raw_probs[row_idx * 2]),
                },
                "b_to_a": {
                    "label": ba_label,
                    "probability": float(raw_probs[row_idx * 2 + 1]),
                },
                "predicted_label": relation_from_minicheck(ab_label, ba_label),
                "note": "MiniCheck is binary entailment; it does not predict contradiction or not relevant.",
            }
        )
    return predictions


def compute_metrics(output_rows: Sequence[Dict[str, Any]], model_key: str) -> Dict[str, Any]:
    total = 0
    per_label_total: Counter = Counter()
    per_label_correct: Counter = Counter()
    supported_labels = SUPPORTED_LABELS[model_key]
    skipped = 0

    for row in output_rows:
        gold = row.get("gold_label", INVALID_LABEL)
        if gold not in supported_labels:
            skipped += 1
            continue
        prediction = row.get("predictions", {}).get(model_key, {})
        if prediction.get("skipped"):
            skipped += 1
            continue
        pred = prediction.get("predicted_label", INVALID_LABEL)
        total += 1
        per_label_total[gold] += 1
        per_label_correct[gold] += int(gold == pred)

    per_label_accuracy: Dict[str, Dict[str, Any]] = {}
    for label in sorted(supported_labels):
        label_total = per_label_total[label]
        label_correct = per_label_correct[label]
        per_label_accuracy[label] = {
            "total": label_total,
            "correct": label_correct,
            "accuracy": label_correct / label_total if label_total else 0.0,
        }

    return {
        "total": total,
        "skipped": skipped,
        "supported_labels": sorted(supported_labels),
        "per_label_accuracy": per_label_accuracy,
    }


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path, args.limit)
    output_rows = [base_row_output(row, idx) for idx, row in enumerate(rows)]

    if "deberta" in args.models:
        indices = supported_row_indices(output_rows, "deberta")
        filtered_rows = [rows[idx] for idx in indices]
        print(f"DeBERTa supported rows: {len(filtered_rows)}/{len(rows)}")
        deberta_predictions = predict_deberta(filtered_rows, args.deberta_model, args.batch_size, args.device)
        for idx, prediction in zip(indices, deberta_predictions):
            output_rows[idx]["predictions"]["deberta"] = prediction
        for idx, output_row in enumerate(output_rows):
            if idx not in set(indices):
                output_row["predictions"]["deberta"] = skipped_prediction(
                    args.deberta_model,
                    "deberta",
                    output_row.get("gold_label", INVALID_LABEL),
                )

    if "minicheck" in args.models:
        indices = supported_row_indices(output_rows, "minicheck")
        filtered_rows = [rows[idx] for idx in indices]
        print(f"MiniCheck supported rows: {len(filtered_rows)}/{len(rows)}")
        minicheck_predictions = predict_minicheck(
            filtered_rows,
            args.minicheck_model,
            args.minicheck_cache_dir,
            args.minicheck_chunk_size,
        )
        for idx, prediction in zip(indices, minicheck_predictions):
            output_rows[idx]["predictions"]["minicheck"] = prediction
        for idx, output_row in enumerate(output_rows):
            if idx not in set(indices):
                output_row["predictions"]["minicheck"] = skipped_prediction(
                    args.minicheck_model,
                    "minicheck",
                    output_row.get("gold_label", INVALID_LABEL),
                )

    metrics = {
        "input_path": os.path.basename(args.input_path),
        "models": args.models,
        "label_set": EVAL_LABELS,
        "supported_labels": {
            model_key: sorted(SUPPORTED_LABELS[model_key])
            for model_key in args.models
        },
        "notes": {
            "deberta": "Only rows whose gold labels are aligned, A>B, B>A, or contradicted are scored. not relevant and A<>B are skipped.",
            "minicheck": "Only rows whose gold labels are aligned, A>B, or B>A are scored. contradicted, not relevant, and A<>B are skipped.",
        },
        "metrics": {
            model_key: compute_metrics(output_rows, model_key)
            for model_key in args.models
        },
    }

    output_jsonl = os.path.join(args.results_dir, os.path.basename(args.output_jsonl))
    metrics_json = os.path.join(args.results_dir, os.path.basename(args.metrics_json))
    write_jsonl(output_jsonl, output_rows)
    write_json(metrics_json, metrics)
    print(f"Saved predictions: {output_jsonl}")
    print(f"Saved metrics: {metrics_json}")


if __name__ == "__main__":
    main()
