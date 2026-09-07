#!/usr/bin/env python3
"""Compute Claim A/B embedding similarity by label type.

Default input:
    datasets/CLAW_4L_RC.jsonl

Default models:
    BAAI/bge-large-en-v1.5
    sentence-transformers/all-mpnet-base-v2
"""

import argparse
import json
import math
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EVAL_DIR)
DEFAULT_INPUT_PATH = os.path.join(REPO_DIR, "datasets", "CLAW_4L_RC.jsonl")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_DIR, "results", "alignment_eval", "embedding_claim_similarity")
DEFAULT_OUTPUT_JSON = "claw4l_rc_embedding_similarity_summary.json"
DEFAULT_MODELS = [
    "BAAI/bge-large-en-v1.5",
    "sentence-transformers/all-mpnet-base-v2",
]
TABLE_MODEL_COLUMNS = [
    ("BAAI/bge-large-en-v1.5", "BGE"),
    ("sentence-transformers/all-mpnet-base-v2", "MPNet"),
]
TABLE_LABEL_ROWS = [
    (r"Aligned -- Exact ($A=B$)", "aligned"),
    (r"Aligned -- Partial ($A > B$)", "partial aligned (A>B)"),
    (r"Aligned -- Partial ($B > A$)", "partial aligned (B>A)"),
    (r"Aligned -- Partial ($A \leftrightarrow B$)", "partial aligned (A<>B)"),
]
TABLE_ALIGNED_LABELS = [label for _display, label in TABLE_LABEL_ROWS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute cosine similarity between reviewed_claim_a.claim and reviewed_claim_b.claim, grouped by label_type."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Input JSONL containing label_type, reviewed_claim_a.claim, and reviewed_claim_b.claim.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the embedding similarity summary JSON file.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help="Output summary JSON file name.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Embedding device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to SentenceTransformer.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face/SentenceTransformers cache directory for downloaded embedding models.",
    )
    return parser.parse_args()


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
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


def get_claim_pair(row: Dict[str, Any]) -> Tuple[str, str]:
    claim_a = row.get("reviewed_claim_a") or {}
    claim_b = row.get("reviewed_claim_b") or {}
    if not isinstance(claim_a, dict):
        claim_a = {}
    if not isinstance(claim_b, dict):
        claim_b = {}
    return safe_text(claim_a.get("claim")), safe_text(claim_b.get("claim"))


def iter_texts_for_embedding(rows: Sequence[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    row_indices: List[int] = []
    for idx, row in enumerate(rows):
        claim_a, claim_b = get_claim_pair(row)
        if not claim_a or not claim_b:
            continue
        texts.extend([claim_a, claim_b])
        row_indices.append(idx)
    return texts, row_indices


def import_sentence_transformer() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: sentence-transformers. Install it in the project environment, for example:\n"
            "  pip install sentence-transformers\n"
            "The models also require torch/transformers."
        ) from exc
    return SentenceTransformer


def load_model(model_name: str, device: Optional[str], trust_remote_code: bool, hf_cache_dir: Optional[str]) -> Any:
    SentenceTransformer = import_sentence_transformer()
    kwargs: Dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    if trust_remote_code:
        kwargs["trust_remote_code"] = True
    if hf_cache_dir:
        kwargs["cache_folder"] = hf_cache_dir
    return SentenceTransformer(model_name, **kwargs)


def encode_normalized(model: Any, texts: Sequence[str], batch_size: int) -> Any:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def dot_similarity(vec_a: Any, vec_b: Any) -> float:
    return float((vec_a * vec_b).sum())


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    clean_values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean_values:
        return {
            "count": 0,
            "mean": None,
        }
    return {
        "count": len(clean_values),
        "mean": float(statistics.fmean(clean_values)),
    }


def compute_model_similarities(
    rows: Sequence[Dict[str, Any]],
    model_name: str,
    batch_size: int,
    device: Optional[str],
    trust_remote_code: bool,
    hf_cache_dir: Optional[str],
) -> Dict[int, float]:
    texts, valid_row_indices = iter_texts_for_embedding(rows)
    if not texts:
        return {}

    print(f"Loading Hugging Face model: {model_name}")
    model = load_model(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        hf_cache_dir=hf_cache_dir,
    )
    embeddings = encode_normalized(model, texts, batch_size=batch_size)

    similarities: Dict[int, float] = {}
    for pair_idx, row_idx in enumerate(valid_row_indices):
        emb_a = embeddings[pair_idx * 2]
        emb_b = embeddings[pair_idx * 2 + 1]
        similarities[row_idx] = dot_similarity(emb_a, emb_b)
    return similarities


def build_row_output(
    rows: Sequence[Dict[str, Any]],
    similarities: Dict[int, float],
    model_name: str,
) -> List[Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        claim_a, claim_b = get_claim_pair(row)
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        out = {
            "row_index": row.get("row_index", idx),
            "qid": row.get("qid", meta.get("qid", "")),
            "label_type": safe_text(row.get("label_type")),
            "target_lang": safe_text(meta.get("target_lang")),
            "country_group": safe_text(meta.get("country_group")),
            "model": model_name,
            "claim_a": claim_a,
            "claim_b": claim_b,
            "cosine_similarity": similarities.get(idx),
        }
        output_rows.append(out)
    return output_rows


def build_summary(row_outputs: Sequence[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for row in row_outputs:
        sim = row.get("cosine_similarity")
        if sim is None:
            continue
        grouped[safe_text(row.get("label_type"))].append(float(sim))

    summary_rows: List[Dict[str, Any]] = []
    for label_type in sorted(grouped):
        stats = summarize(grouped[label_type])
        summary_rows.append({"model": model_name, "label_type": label_type, **stats})

    all_values = [float(row["cosine_similarity"]) for row in row_outputs if row.get("cosine_similarity") is not None]
    summary_rows.append({"model": model_name, "label_type": "__all__", **summarize(all_values)})
    return summary_rows


def summary_index(summary_rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        (row["model"], row["label_type"]): row
        for row in summary_rows
    }


def count_for_label(index: Dict[Tuple[str, str], Dict[str, Any]], label: str) -> int:
    first_model = TABLE_MODEL_COLUMNS[0][0]
    return int(index[(first_model, label)]["count"])


def mean_percent_value(index: Dict[Tuple[str, str], Dict[str, Any]], model: str, label: str) -> float:
    return round(float(index[(model, label)]["mean"]) * 100, 2)


def weighted_mean(index: Dict[Tuple[str, str], Dict[str, Any]], model: str, labels: Sequence[str]) -> Tuple[int, float]:
    total = 0
    weighted_sum = 0.0
    for label in labels:
        item = index[(model, label)]
        count = int(item["count"])
        total += count
        weighted_sum += count * float(item["mean"])
    return total, weighted_sum / total if total else 0.0


def build_table(summary_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    index = summary_index(summary_rows)
    rows: List[Dict[str, Any]] = []

    for display, label in TABLE_LABEL_ROWS:
        values = [mean_percent_value(index, model, label) for model, _name in TABLE_MODEL_COLUMNS]
        rows.append(
            {
                "label": display,
                "count": count_for_label(index, label),
                "BGE": values[0],
                "MPNet": values[1],
            }
        )

    aligned_count, aligned_bge = weighted_mean(index, TABLE_MODEL_COLUMNS[0][0], TABLE_ALIGNED_LABELS)
    _aligned_count_mpnet, aligned_mpnet = weighted_mean(index, TABLE_MODEL_COLUMNS[1][0], TABLE_ALIGNED_LABELS)
    rows.append(
        {
            "label": "Aligned",
            "count": aligned_count,
            "BGE": round(aligned_bge * 100, 2),
            "MPNet": round(aligned_mpnet * 100, 2),
            "row_type": "subtotal",
        }
    )

    for display, label in [
        (r"Contradicted ($A \perp B$)", "contradict"),
        (r"Not Relevant ($A \dashv B$)", "not relevant"),
    ]:
        values = [mean_percent_value(index, model, label) for model, _name in TABLE_MODEL_COLUMNS]
        rows.append(
            {
                "label": display,
                "count": count_for_label(index, label),
                "BGE": values[0],
                "MPNet": values[1],
            }
        )

    total_values = [mean_percent_value(index, model, "__all__") for model, _name in TABLE_MODEL_COLUMNS]
    total_count = count_for_label(index, "__all__")
    rows.append(
        {
            "label": "Weighted Total",
            "count": total_count,
            "BGE": total_values[0],
            "MPNet": total_values[1],
            "row_type": "total",
        }
    )
    return {
        "caption": "Label distribution and semantic similarity scores in CLAW-4L-RC.",
        "label": "tab:claw4l-rc-label-dist",
        "columns": ["Label", "#", "BGE", "MPNet"],
        "values_are_percentages": True,
        "rows": rows,
    }


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    rows = load_jsonl(args.input_path)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loaded {len(rows)} rows from {args.input_path}")

    all_summary_rows: List[Dict[str, Any]] = []
    for model_name in DEFAULT_MODELS:
        similarities = compute_model_similarities(
            rows=rows,
            model_name=model_name,
            batch_size=args.batch_size,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
            hf_cache_dir=args.hf_cache_dir,
        )
        row_outputs = build_row_output(
            rows=rows,
            similarities=similarities,
            model_name=model_name,
        )
        summary_rows = build_summary(row_outputs, model_name=model_name)
        all_summary_rows.extend(summary_rows)

    output = {
        "input_path": os.path.basename(args.input_path),
        "models": DEFAULT_MODELS,
        "label_type_summary": all_summary_rows,
        "table": build_table(all_summary_rows),
    }

    combined_json_path = os.path.join(args.output_dir, os.path.basename(args.output_json))
    write_json(combined_json_path, output)
    print(f"Wrote combined summary JSON: {combined_json_path}")


if __name__ == "__main__":
    main()
