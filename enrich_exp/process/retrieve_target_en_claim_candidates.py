#!/usr/bin/env python3
"""Retrieve EN claim candidates for each target-language claim within each biography sample."""

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENRICH_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(ENRICH_DIR)
DEFAULT_EN_CLAIMS_JSONL = os.path.join(
    PROJECT_DIR,
    "datasets",
    "pool_300_en_claims_gpt_5_1.jsonl",
)
DEFAULT_TARGET_CLAIMS_JSONL = os.path.join(
    PROJECT_DIR,
    "datasets",
    "pool_300_target_claims_gpt_5_1.jsonl",
)
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "results", "enrich_exp", "enrich_claim_similarity")
DEFAULT_OUTPUT_JSONL = "target_to_en_claim_top5_mpnet.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each biography sample, retrieve top-k English claims for each target-language claim "
            "using sentence-transformers/all-mpnet-base-v2 cosine similarity."
        )
    )
    parser.add_argument("--en-claims-jsonl", type=str, default=DEFAULT_EN_CLAIMS_JSONL)
    parser.add_argument("--target-claims-jsonl", type=str, default=DEFAULT_TARGET_CLAIMS_JSONL)
    parser.add_argument("--results-dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-jsonl", type=str, default=DEFAULT_OUTPUT_JSONL)
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None, help="Optional sentence-transformers device, e.g. cuda or cpu.")
    parser.add_argument("--limit", type=int, default=None, help="Limit joined samples for debugging.")
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


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def collect_claims(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    for sent_item in row.get("decomposition_results", []) or []:
        for claim_item in sent_item.get("claims", []) or []:
            claim_text = safe_text(claim_item.get("claim"))
            if not claim_text:
                continue
            claims.append(
                {
                    "claim_idx": len(claims),
                    "claim": claim_text,
                }
            )
    return claims


def row_meta(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "qid": row.get("qid"),
        "name": row.get("name"),
        "country_group": row.get("country_group"),
        "occupations": row.get("occupations"),
        "target_lang": row.get("target_lang"),
        "en_title": row.get("en_title"),
        "target_title": row.get("target_title"),
    }


def display_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, PROJECT_DIR)
    except ValueError:
        return path


def join_rows(en_rows: Sequence[Dict[str, Any]], target_rows: Sequence[Dict[str, Any]], limit: Any) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    en_by_qid = {row.get("qid"): row for row in en_rows}
    joined: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for target_row in target_rows:
        qid = target_row.get("qid")
        en_row = en_by_qid.get(qid)
        if en_row is None:
            continue
        joined.append((en_row, target_row))
        if limit is not None and len(joined) >= limit:
            break
    return joined


def encode_claims(model: Any, claims: Sequence[Dict[str, Any]], batch_size: int) -> Any:
    texts = [claim["claim"] for claim in claims]
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def retrieve_for_sample(
    model: Any,
    en_claims: Sequence[Dict[str, Any]],
    target_claims: Sequence[Dict[str, Any]],
    top_k: int,
    threshold: float,
    batch_size: int,
) -> List[Dict[str, Any]]:
    import torch

    if not target_claims:
        return []
    if not en_claims:
        return [
            {
                "target_claim_idx": target_claim["claim_idx"],
                "target_claim": target_claim["claim"],
                "candidates": [],
            }
            for target_claim in target_claims
        ]

    en_embeddings = encode_claims(model, en_claims, batch_size)
    target_embeddings = encode_claims(model, target_claims, batch_size)
    scores = target_embeddings @ en_embeddings.T
    k = min(top_k, len(en_claims))
    top_scores, top_indices = torch.topk(scores, k=k, dim=1)

    outputs: List[Dict[str, Any]] = []
    for target_claim, score_row, index_row in zip(target_claims, top_scores, top_indices):
        candidates = []
        for score, en_idx_tensor in zip(score_row.tolist(), index_row.tolist()):
            if score < threshold:
                continue
            en_claim = en_claims[int(en_idx_tensor)]
            candidates.append(
                {
                    "en_claim_idx": en_claim["claim_idx"],
                    "en_claim": en_claim["claim"],
                    "similarity": round(float(score), 6),
                }
            )
        outputs.append(
            {
                "target_claim_idx": target_claim["claim_idx"],
                "target_claim": target_claim["claim"],
                "candidates": candidates,
            }
        )
    return outputs


def build_outputs(args: argparse.Namespace) -> List[Dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    en_rows = read_jsonl(args.en_claims_jsonl)
    target_rows = read_jsonl(args.target_claims_jsonl)
    joined_rows = join_rows(en_rows, target_rows, args.limit)
    if args.device:
        model = SentenceTransformer(args.model_name, device=args.device)
    else:
        model = SentenceTransformer(args.model_name)

    outputs: List[Dict[str, Any]] = []
    for sample_idx, (en_row, target_row) in enumerate(joined_rows, start=1):
        en_claims = collect_claims(en_row)
        target_claims = collect_claims(target_row)
        retrieved = retrieve_for_sample(
            model,
            en_claims,
            target_claims,
            args.top_k,
            args.threshold,
            args.batch_size,
        )
        outputs.append(
            {
                "sample_idx": sample_idx - 1,
                "qid": target_row.get("qid"),
                "meta": row_meta(target_row),
                "en_claim_count": len(en_claims),
                "target_claim_count": len(target_claims),
                "target_to_en_candidates": retrieved,
            }
        )
        print(
            f"[{sample_idx}/{len(joined_rows)}] qid={target_row.get('qid')} "
            f"target_claims={len(target_claims)} en_claims={len(en_claims)} "
            f"candidate_links={sum(len(item['candidates']) for item in retrieved)}",
            flush=True,
        )
    return outputs


def main() -> None:
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)
    output_jsonl = resolve_output_path(results_dir, args.output_jsonl)
    outputs = build_outputs(args)
    write_jsonl(output_jsonl, outputs)
    print(f"Saved retrieval output: {output_jsonl}")


if __name__ == "__main__":
    main()
