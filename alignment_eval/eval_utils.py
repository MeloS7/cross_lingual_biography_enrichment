#!/usr/bin/env python3
"""Shared utilities for claim alignment evaluation pipelines."""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import yaml


EVAL_LABELS = [
    "aligned",
    "contradicted",
    "not relevant",
    "partial aligned (A>B)",
    "partial aligned (B>A)",
    "partial aligned (A<>B)",
]
INVALID_LABEL = "__invalid__"
ALIGNMENT_CHOICES = ["Aligned", "Contradicted", "Not Relevant"]
ENRICHMENT_CHOICES = ["None", "A > B", "B > A", "A <> B"]


EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(EVAL_DIR)
DEFAULT_PROMPTS_YAML = os.path.join(REPO_DIR, "prompts", "claim_alignment_prompts_claim_only.yaml")
DEFAULT_RESULTS_DIR = os.path.join(REPO_DIR, "results", "alignment_eval")


def load_prompts(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    system_prompt = data.get("SYSTEM PROMPT") or data.get("system_prompt")
    user_prompt_template = data.get("USER PROMPT") or data.get("user_prompt_template")

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(f"Missing or invalid 'SYSTEM PROMPT' in {path}")
    if not isinstance(user_prompt_template, str) or not user_prompt_template.strip():
        raise ValueError(f"Missing or invalid 'USER PROMPT' in {path}")

    return system_prompt.rstrip(), user_prompt_template.rstrip()


def safe_get(d: Dict[str, Any], key: str) -> str:
    v = d.get(key, "")
    if v is None:
        return ""
    text = str(v).strip()
    return text


def get_row_value(row: Dict[str, Any], key: str, default: Any = "") -> Any:
    meta = row.get("meta")
    if isinstance(meta, dict) and key in meta:
        return meta.get(key)
    return default


def build_user_prompt(
    row: Dict[str, Any],
    claim_a: Dict[str, Any],
    claim_b: Dict[str, Any],
    user_prompt_template: str,
) -> str:
    values = {
        "name": safe_get({"name": get_row_value(row, "name")}, "name"),
        "en_title": safe_get({"en_title": get_row_value(row, "en_title")}, "en_title"),
        "target_title": safe_get({"target_title": get_row_value(row, "target_title")}, "target_title"),
        "a_claim": safe_get(claim_a, "claim"),
        "b_claim": safe_get(claim_b, "claim"),
        "a_subject": safe_get(claim_a, "subject"),
        "a_predicate": safe_get(claim_a, "predicate"),
        "a_object": safe_get(claim_a, "object"),
        "a_time": safe_get(claim_a, "time"),
        "a_location": safe_get(claim_a, "location"),
        "a_reason": safe_get(claim_a, "reason"),
        "a_manner": safe_get(claim_a, "manner"),
        "a_hedge": safe_get(claim_a, "hedge"),
        "a_source_sent": safe_get(claim_a, "source_sent"),
        "b_subject": safe_get(claim_b, "subject"),
        "b_predicate": safe_get(claim_b, "predicate"),
        "b_object": safe_get(claim_b, "object"),
        "b_time": safe_get(claim_b, "time"),
        "b_location": safe_get(claim_b, "location"),
        "b_reason": safe_get(claim_b, "reason"),
        "b_manner": safe_get(claim_b, "manner"),
        "b_hedge": safe_get(claim_b, "hedge"),
        "b_source_sent": safe_get(claim_b, "source_sent"),
    }
    return user_prompt_template.format(**values)


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_results_path(results_dir: str, requested_path: str) -> str:
    filename = os.path.basename(str(requested_path).strip())
    if not filename:
        raise ValueError(f"Invalid output file name/path: {requested_path}")
    return os.path.join(results_dir, filename)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def canonicalize_label(raw_label: Any) -> str:
    if raw_label is None:
        return INVALID_LABEL
    text = str(raw_label).strip()
    if not text:
        return INVALID_LABEL

    for label in EVAL_LABELS:
        if text == label:
            return label

    norm = normalize_text(text)
    alias_patterns: List[Tuple[str, str]] = [
        (r"partial\s*aligned\s*\(\s*a\s*<\s*>\s*b\s*\)", "partial aligned (A<>B)"),
        (r"partial\s*aligned\s*\(\s*a\s*>\s*b\s*\)", "partial aligned (A>B)"),
        (r"partial\s*aligned\s*\(\s*b\s*>\s*a\s*\)", "partial aligned (B>A)"),
        (r"\bcontradict(?:ed|ion)?\b", "contradicted"),
        (r"\bnot\s*relevant\b", "not relevant"),
        (r"\baligned\b", "aligned"),
    ]
    for pattern, label in alias_patterns:
        if re.search(pattern, norm, flags=re.IGNORECASE):
            return label
    return INVALID_LABEL


def canonicalize_alignment(raw_alignment: Any) -> str:
    if raw_alignment is None:
        return ""
    text = str(raw_alignment).strip()
    if not text:
        return ""
    norm = normalize_text(text)
    if re.search(r"\bnot\s*relevant\b|\birrelevant\b", norm, flags=re.IGNORECASE):
        return "Not Relevant"
    if re.search(r"\bcontradicted\b|\bcontradiction\b", norm, flags=re.IGNORECASE):
        return "Contradicted"
    if re.search(r"\baligned\b|\bequivalent\b", norm, flags=re.IGNORECASE):
        return "Aligned"
    return ""


def canonicalize_enrichment(raw_enrichment: Any) -> str:
    if raw_enrichment is None:
        return ""
    text = str(raw_enrichment).strip()
    if not text:
        return ""
    norm = normalize_text(text)

    if re.search(r"^none$|\bno\s*enrichment\b", norm, flags=re.IGNORECASE):
        return "None"
    if re.search(r"\ba\s*<\s*>\s*b\b|\ba\s*<->\s*b\b", norm, flags=re.IGNORECASE):
        return "A <> B"
    if re.search(r"\ba\s*(>|->)\s*b\b|\ba\s*>\s*b\b", norm, flags=re.IGNORECASE):
        return "A > B"
    if re.search(r"\bb\s*(>|->)\s*a\b|\bb\s*>\s*a\b", norm, flags=re.IGNORECASE):
        return "B > A"
    return ""


def strip_generation_wrappers(raw_text: str) -> str:
    if raw_text is None:
        return ""
    text = raw_text.strip()
    if not text:
        return ""

    # If any model still emits internal-thought tags, keep only the trailing segment.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()

    text = re.sub(r"^\s*final\s*answer\s*[:：]\s*", "", text, flags=re.IGNORECASE).strip()

    fenced = re.search(r"```(?:json|JSON)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    return text


def extract_final_answer(raw_text: str) -> str:
    return strip_generation_wrappers(raw_text)


def parse_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    candidates = [text]
    matched = re.search(r"\{[\s\S]*\}", text)
    if matched:
        candidates.insert(0, matched.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def five_label_to_structured(label: str) -> Tuple[str, str]:
    if label == "aligned":
        return "Aligned", "None"
    if label == "contradicted":
        return "Contradicted", ""
    if label == "not relevant":
        return "Not Relevant", ""
    if label == "partial aligned (A>B)":
        return "Aligned", "A > B"
    if label == "partial aligned (B>A)":
        return "Aligned", "B > A"
    if label == "partial aligned (A<>B)":
        return "Aligned", "A <> B"
    return "", ""


def structured_to_eval_label(alignment: str, enrichment: str) -> str:
    if alignment == "Contradicted":
        return "contradicted"
    if alignment == "Not Relevant":
        return "not relevant"
    if alignment != "Aligned":
        return INVALID_LABEL

    if enrichment == "":
        return "aligned"
    if enrichment == "None":
        return "aligned"
    if enrichment == "A > B":
        return "partial aligned (A>B)"
    if enrichment == "B > A":
        return "partial aligned (B>A)"
    if enrichment == "A <> B":
        return "partial aligned (A<>B)"
    return INVALID_LABEL


def extract_structured_prediction(raw_text: str) -> Tuple[str, str, str, str]:
    text = extract_final_answer(raw_text)
    if not text:
        return "", "", INVALID_LABEL, ""

    parsed = parse_json_object_from_text(text)
    alignment = ""
    enrichment = ""
    why = ""

    if isinstance(parsed, dict):
        raw_why = parsed.get("Why", parsed.get("why", ""))
        if raw_why is not None:
            why = str(raw_why).strip()
        alignment = canonicalize_alignment(parsed.get("Alignment", parsed.get("alignment", "")))
        enrichment = canonicalize_enrichment(parsed.get("Enrichment", parsed.get("enrichment", "")))
        if not alignment:
            legacy_label = canonicalize_label(parsed.get("label", ""))
            if legacy_label != INVALID_LABEL:
                alignment, enrichment = five_label_to_structured(legacy_label)
    else:
        # Backward-compatible fallback for plain 5-label text output.
        legacy_label = canonicalize_label(text)
        if legacy_label != INVALID_LABEL:
            alignment, enrichment = five_label_to_structured(legacy_label)

    # Some smaller models output empty enrichment for aligned cases.
    # Treat empty as "None" to avoid penalizing format-only omissions.
    if alignment == "Aligned" and enrichment == "":
        enrichment = "None"

    if alignment != "Aligned":
        enrichment = ""

    pred_label = structured_to_eval_label(alignment, enrichment)
    return alignment, enrichment, pred_label, why


def get_claim_pair(row: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    reviewed_a = row.get("reviewed_claim_a")
    reviewed_b = row.get("reviewed_claim_b")
    if isinstance(reviewed_a, dict) and isinstance(reviewed_b, dict):
        return reviewed_a, reviewed_b

    raise ValueError(
        "Missing claim pair in row: expected "
        "(reviewed_claim_a, reviewed_claim_b)."
    )


def get_gold_label(row: Dict[str, Any]) -> str:
    return canonicalize_label(row.get("label_type"))


def compute_metrics(pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eval_rows = [x for x in pred_rows if x.get("gold_label") in EVAL_LABELS]
    total = len(eval_rows)
    skipped_no_gold = len(pred_rows) - total
    correct = sum(1 for x in eval_rows if x["pred_label"] == x["gold_label"])
    accuracy = correct / total if total else 0.0

    confusion: Dict[str, Dict[str, int]] = {g: {p: 0 for p in EVAL_LABELS + [INVALID_LABEL]} for g in EVAL_LABELS}
    for row in eval_rows:
        g = row.get("gold_label")
        p = row.get("pred_label", INVALID_LABEL)
        if g not in confusion:
            continue
        if p not in confusion[g]:
            p = INVALID_LABEL
        confusion[g][p] += 1

    f1_values = []
    for label in EVAL_LABELS:
        tp = confusion[label].get(label, 0)
        fp = sum(confusion[g].get(label, 0) for g in EVAL_LABELS if g != label)
        fn = sum(confusion[label].get(p, 0) for p in EVAL_LABELS + [INVALID_LABEL] if p != label)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1_values.append(f1)

    metrics = {
        "accuracy": accuracy,
        "f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "total_rows": len(pred_rows),
        "evaluated_rows": total,
        "skipped_rows_without_valid_gold_label": skipped_no_gold,
        "confusion_matrix": confusion,
    }
    return metrics
