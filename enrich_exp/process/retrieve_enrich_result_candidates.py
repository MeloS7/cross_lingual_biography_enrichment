#!/usr/bin/env python3
"""Retrieve before/after enrichment claim candidates with cosine similarity."""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENRICH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(ENRICH_DIR)
# Change these values before running on a different after-enrichment file.
METHOD = "target_enrich"
TARGET_LANG = "zh"
MODEL_NAME = "gemma4"

DEFAULT_BEFORE_EN_JSONL = os.path.join(
    PROJECT_DIR,
    "datasets",
    "pool_300_en_claims_gpt_5_1.jsonl",
)
DEFAULT_BEFORE_TARGET_JSONL = os.path.join(
    PROJECT_DIR,
    "datasets",
    "pool_300_target_claims_gpt_5_1.jsonl",
)
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_BEFORE_FIELD = "before_enrich"
DEFAULT_AFTER_FIELD = "after_enrich"


def default_after_jsonl(method: str, target_lang: str, model_name: str) -> str:
    return os.path.join(
        PROJECT_DIR,
        "datasets",
        "claim_enrich",
        method,
        target_lang,
        model_name,
        "after_enrich_claims.jsonl",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join before-enrichment claim files by qid, compare them with after-enrichment "
            "claims, and keep either all bidirectional similarities or top-k candidates above "
            "the cosine similarity threshold."
        )
    )
    parser.add_argument("--before-en-jsonl", type=str, default=DEFAULT_BEFORE_EN_JSONL)
    parser.add_argument("--before-target-jsonl", type=str, default=DEFAULT_BEFORE_TARGET_JSONL)
    parser.add_argument("--after-jsonl", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--method", type=str, default=METHOD)
    parser.add_argument("--target-lang", type=str, default=TARGET_LANG)
    parser.add_argument("--model-name-dir", type=str, default=MODEL_NAME)
    parser.add_argument("--model-name", type=str, default=DEFAULT_EMBEDDING_MODEL_NAME)
    parser.add_argument("--device", type=str, default=None, help="SentenceTransformer device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Force loading embedding model/tokenizer from local files only.",
    )
    parser.add_argument("--before-field", type=str, default=DEFAULT_BEFORE_FIELD)
    parser.add_argument("--after-field", type=str, default=DEFAULT_AFTER_FIELD)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--all-similarities",
        action="store_true",
        help="Save all before/after claim similarity scores, ignoring --top-k and --threshold.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--details-jsonl", type=str, default="before_after_enrich_similarity.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.after_jsonl is None:
        args.after_jsonl = default_after_jsonl(args.method, args.target_lang, args.model_name_dir)
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.after_jsonl)
    return args


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_input_idx"] = idx
            rows.append(row)
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def collect_claim_texts(row: Dict[str, Any]) -> List[str]:
    claims: List[str] = []
    for sent_item in row.get("decomposition_results", []) or []:
        for claim_item in sent_item.get("claims", []) or []:
            claim_text = safe_text(claim_item.get("claim"))
            if claim_text:
                claims.append(claim_text)
    return claims


def merge_claim_lists(claim_lists: Sequence[Sequence[str]]) -> List[str]:
    seen = set()
    merged: List[str] = []
    for claims in claim_lists:
        for claim in claims:
            if claim not in seen:
                seen.add(claim)
                merged.append(claim)
    return merged


def build_joined_rows(
    before_en_rows: Sequence[Dict[str, Any]],
    before_target_rows: Sequence[Dict[str, Any]],
    after_rows: Sequence[Dict[str, Any]],
    before_field: str,
    after_field: str,
    limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    before_en_by_qid = {row.get("qid"): row for row in before_en_rows}
    before_target_by_qid = {row.get("qid"): row for row in before_target_rows}
    joined_rows: List[Dict[str, Any]] = []
    missing_before_en = 0
    missing_before_target = 0

    for after_row in after_rows:
        qid = after_row.get("qid")
        before_en_row = before_en_by_qid.get(qid)
        before_target_row = before_target_by_qid.get(qid)
        if before_en_row is None:
            missing_before_en += 1
            continue
        if before_target_row is None:
            missing_before_target += 1
            continue

        before_en_claims = collect_claim_texts(before_en_row)
        before_target_claims = collect_claim_texts(before_target_row)
        after_claims = collect_claim_texts(after_row)
        before_claims = merge_claim_lists([before_en_claims, before_target_claims])

        joined_rows.append(
            {
                "_input_idx": after_row.get("_input_idx"),
                "qid": qid,
                "name": after_row.get("name"),
                "country_group": after_row.get("country_group"),
                "occupations": after_row.get("occupations"),
                "target_lang": after_row.get("target_lang"),
                "en_title": after_row.get("en_title"),
                "target_title": after_row.get("target_title"),
                "before_en_claim_count": len(before_en_claims),
                "before_target_claim_count": len(before_target_claims),
                "claims": {
                    before_field: before_claims,
                    after_field: after_claims,
                },
            }
        )
        if limit is not None and len(joined_rows) >= limit:
            break

    join_summary = {
        "after_row_count": len(after_rows),
        "before_en_row_count": len(before_en_rows),
        "before_target_row_count": len(before_target_rows),
        "joined_row_count": len(joined_rows),
        "missing_before_en_count": missing_before_en,
        "missing_before_target_count": missing_before_target,
    }
    return joined_rows, join_summary


def get_claims(row: Dict[str, Any], field: str) -> List[str]:
    claims = row.get("claims", {}).get(field, [])
    if not isinstance(claims, list):
        raise ValueError(f"Expected claims[{field!r}] to be a list for qid={row.get('qid')}")
    return [str(claim).strip() for claim in claims if str(claim).strip()]


def collect_unique_claims(rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> List[str]:
    seen = set()
    unique_claims: List[str] = []
    for row in rows:
        for field in fields:
            for claim in get_claims(row, field):
                if claim not in seen:
                    seen.add(claim)
                    unique_claims.append(claim)
    return unique_claims


def load_sentence_transformer(
    model_name: str,
    device: Optional[str] = None,
    local_files_only: bool = False,
) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: sentence-transformers. "
            "Install with `pip install sentence-transformers`."
        ) from exc
    kwargs: Dict[str, Any] = {}
    if device:
        kwargs["device"] = device
    if local_files_only:
        kwargs["local_files_only"] = True
    try:
        return SentenceTransformer(model_name, **kwargs)
    except TypeError:
        # Backward compatibility for older sentence-transformers without local_files_only.
        kwargs.pop("local_files_only", None)
        return SentenceTransformer(model_name, **kwargs)


def build_embedding_cache(claims: Sequence[str], model: Any, batch_size: int) -> Dict[str, np.ndarray]:
    embeddings = model.encode(
        list(claims),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return {claim: embeddings[idx] for idx, claim in enumerate(claims)}


def matrix_for_claims(claims: Sequence[str], embedding_cache: Dict[str, np.ndarray]) -> np.ndarray:
    if not claims:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack([embedding_cache[claim] for claim in claims])


def top_indices_above_threshold(row_scores: np.ndarray, threshold: float, top_k: int) -> np.ndarray:
    candidate_indices = np.flatnonzero(row_scores > threshold)
    effective_top_k = min(top_k, len(candidate_indices))
    if not effective_top_k:
        return np.array([], dtype=np.int64)

    candidate_scores = row_scores[candidate_indices]
    top_candidate_positions = np.argpartition(
        -candidate_scores,
        range(effective_top_k),
    )[:effective_top_k]
    top_candidate_positions = top_candidate_positions[
        np.argsort(-candidate_scores[top_candidate_positions])
    ]
    return candidate_indices[top_candidate_positions]


def all_indices_by_score(row_scores: np.ndarray) -> np.ndarray:
    return np.argsort(-row_scores)


def compute_similarity_matrix(
    left_claims: Sequence[str],
    right_claims: Sequence[str],
    embedding_cache: Dict[str, np.ndarray],
) -> np.ndarray:
    if not left_claims or not right_claims:
        return np.empty((len(left_claims), len(right_claims)), dtype=np.float32)

    left_matrix = matrix_for_claims(left_claims, embedding_cache)
    right_matrix = matrix_for_claims(right_claims, embedding_cache)
    return left_matrix @ right_matrix.T


def before_to_after_matches(
    before_claims: Sequence[str],
    after_claims: Sequence[str],
    similarities: np.ndarray,
    top_k: int,
    threshold: float,
    all_similarities: bool,
) -> List[List[Dict[str, Any]]]:
    if not all_similarities and top_k <= 0:
        raise ValueError("--top-k must be positive.")

    if not before_claims:
        return []
    if not after_claims:
        return [[] for _ in before_claims]

    matches: List[List[Dict[str, Any]]] = []
    for before_idx, _before_claim in enumerate(before_claims):
        row_scores = similarities[before_idx]
        if all_similarities:
            candidate_indices = all_indices_by_score(row_scores)
        else:
            candidate_indices = top_indices_above_threshold(row_scores, threshold, top_k)
        matches.append(
            [
                {
                    "after_idx": int(after_idx),
                    "after_claim": after_claims[int(after_idx)],
                    "after_similarity": float(row_scores[int(after_idx)]),
                }
                for after_idx in candidate_indices
            ]
        )
    return matches


def after_to_before_matches(
    before_claims: Sequence[str],
    after_claims: Sequence[str],
    similarities: np.ndarray,
    top_k: int,
    threshold: float,
    all_similarities: bool,
) -> List[Dict[str, Any]]:
    if not all_similarities and top_k <= 0:
        raise ValueError("--top-k must be positive.")

    results: List[Dict[str, Any]] = []
    for after_idx, after_claim in enumerate(after_claims):
        row_scores = similarities[:, after_idx] if len(before_claims) else np.array([], dtype=np.float32)
        if all_similarities:
            candidate_indices = all_indices_by_score(row_scores)
        else:
            candidate_indices = top_indices_above_threshold(row_scores, threshold, top_k)
        results.append(
            {
                "after_idx": after_idx,
                "after_claim": after_claim,
                "before_matches": [
                    {
                        "before_idx": int(before_idx),
                        "before_claim": before_claims[int(before_idx)],
                        "before_similarity": float(row_scores[int(before_idx)]),
                    }
                    for before_idx in candidate_indices
                ],
            }
        )
    return results


def evaluate_rows(
    rows: Sequence[Dict[str, Any]],
    before_field: str,
    after_field: str,
    embedding_cache: Dict[str, np.ndarray],
    top_k: int,
    threshold: float,
    all_similarities: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    detail_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "sample_count": len(rows),
        "total_before_claims": 0,
        "total_after_claims": 0,
        "before_claims_with_matches": 0,
        "after_claims_with_matches": 0,
        "total_before_to_after_matches": 0,
        "total_after_to_before_matches": 0,
    }

    for row in rows:
        before_claims = get_claims(row, before_field)
        after_claims = get_claims(row, after_field)
        similarities = compute_similarity_matrix(before_claims, after_claims, embedding_cache)
        if all_similarities:
            before_matches = [[] for _ in before_claims]
        else:
            before_matches = before_to_after_matches(
                before_claims,
                after_claims,
                similarities,
                top_k=top_k,
                threshold=threshold,
                all_similarities=all_similarities,
            )
        after_matches = after_to_before_matches(
            before_claims,
            after_claims,
            similarities,
            top_k=top_k,
            threshold=threshold,
            all_similarities=all_similarities,
        )

        summary["total_before_claims"] += len(before_claims)
        summary["total_after_claims"] += len(after_claims)
        summary["before_claims_with_matches"] += sum(1 for matches in before_matches if matches)
        summary["after_claims_with_matches"] += sum(1 for item in after_matches if item["before_matches"])
        summary["total_before_to_after_matches"] += sum(len(matches) for matches in before_matches)
        summary["total_after_to_before_matches"] += sum(len(item["before_matches"]) for item in after_matches)

        detail_rows.append(
            {
                "input_idx": row.get("_input_idx"),
                "qid": row.get("qid"),
                "name": row.get("name"),
                "en_title": row.get("en_title"),
                "target_title": row.get("target_title"),
                "country_group": row.get("country_group"),
                "target_lang": row.get("target_lang"),
                "before_field": before_field,
                "after_field": after_field,
                "before_en_claim_count": row.get("before_en_claim_count"),
                "before_target_claim_count": row.get("before_target_claim_count"),
                "before_claim_count": len(before_claims),
                "after_claim_count": len(after_claims),
                "retrieval_mode": "all_similarities" if all_similarities else "topk_threshold",
                "top_k": None if all_similarities else top_k,
                "threshold": None if all_similarities else threshold,
                "before_claims": [
                    {
                        "before_idx": idx,
                        "before_claim": claim,
                        "after_matches": before_matches[idx],
                    }
                    for idx, claim in enumerate(before_claims)
                ] if not all_similarities else [],
                "after_claims": after_matches,
            }
        )

    return detail_rows, summary


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    before_en_rows = read_jsonl(args.before_en_jsonl)
    before_target_rows = read_jsonl(args.before_target_jsonl)
    after_rows = read_jsonl(args.after_jsonl)
    joined_rows, join_summary = build_joined_rows(
        before_en_rows,
        before_target_rows,
        after_rows,
        before_field=args.before_field,
        after_field=args.after_field,
        limit=args.limit,
    )
    if not joined_rows:
        raise ValueError("No joined rows available after qid merge.")

    fields = [args.before_field, args.after_field]
    unique_claims = collect_unique_claims(joined_rows, fields)

    print(f"Loaded joined rows: {len(joined_rows)}")
    print(f"Unique claims to encode: {len(unique_claims)}")
    model = load_sentence_transformer(
        args.model_name,
        device=args.device,
        local_files_only=bool(args.local_files_only),
    )
    embedding_cache = build_embedding_cache(unique_claims, model, args.batch_size)

    detail_rows, eval_summary = evaluate_rows(
        joined_rows,
        before_field=args.before_field,
        after_field=args.after_field,
        embedding_cache=embedding_cache,
        top_k=args.top_k,
        threshold=args.threshold,
        all_similarities=args.all_similarities,
    )

    details_path = os.path.join(args.output_dir, args.details_jsonl)
    write_jsonl(details_path, detail_rows)

    print(f"Saved details: {details_path}")
    print(f"Join summary: {json.dumps(join_summary, ensure_ascii=False)}")
    print(f"Retrieval summary: {json.dumps(eval_summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
