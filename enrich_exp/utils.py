"""Shared utilities for claim alignment verification scripts."""

import argparse
import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROMPTS_YAML = os.path.join(SCRIPT_DIR, "prompts", "claim_alignment_prompts_claim_only.yaml")


def resolve_output_path(results_dir: str, path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(results_dir, path)


def display_path_from_repo(path: str, repo_name: str = "cross_ling_female_bios") -> str:
    """Return a portable path starting at the repository directory when possible."""
    if not path:
        return path
    parts = os.path.abspath(path).split(os.sep)
    try:
        start = parts.index(repo_name)
    except ValueError:
        return path
    return os.path.join(*parts[start:])


def display_paths_from_repo(paths: Iterable[str], repo_name: str = "cross_ling_female_bios") -> List[str]:
    return [display_path_from_repo(path, repo_name=repo_name) for path in paths]


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


def load_prompts(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Prompt YAML must contain a mapping: {path}")
    system_prompt = str(data.get("SYSTEM PROMPT", data.get("system", ""))).strip()
    user_prompt = str(data.get("USER PROMPT", data.get("user", ""))).strip()
    if not user_prompt:
        raise ValueError(f"Prompt YAML must contain a user prompt: {path}")
    return system_prompt, user_prompt


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def render_prompt(template: str, values: Dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", stringify(value))
    return rendered


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _decode_json_string_fragment(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value.replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")


def _normalize_alignment_value(value: str) -> str:
    normalized = " ".join(str(value).split())
    lowered = normalized.casefold()
    if lowered.startswith("align"):
        return "Aligned"
    if lowered.startswith("contrad"):
        return "Contradicted"
    if lowered.startswith("not") and "relevant" in lowered:
        return "Not Relevant"
    return normalized


def _normalize_enrichment_value(value: str) -> str:
    normalized = " ".join(str(value).split())
    lowered = normalized.replace(" ", "").casefold()
    if lowered in {"", "none"}:
        return "None" if lowered == "none" else ""
    if lowered == "a>b":
        return "A > B"
    if lowered == "b>a":
        return "B > A"
    if lowered in {"a<>b", "a< >b", "a<> b"}:
        return "A <> B"
    return normalized


def recover_alignment_payload(text: str) -> Optional[Dict[str, Any]]:
    alignment_match = re.search(r'"Alignment"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    if alignment_match is None:
        return None
    enrichment_match = re.search(r'"Enrichment"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    why_match = re.search(r'"Why"\s*:\s*"((?:\\.|[^"\\])*)"', text, flags=re.DOTALL)
    recovered: Dict[str, Any] = {
        "Alignment": _normalize_alignment_value(_decode_json_string_fragment(alignment_match.group(1))),
        "Enrichment": _normalize_enrichment_value(
            _decode_json_string_fragment(enrichment_match.group(1)) if enrichment_match else ""
        ),
    }
    if why_match is not None:
        recovered["Why"] = _decode_json_string_fragment(why_match.group(1))
    return recovered


def parse_json_response(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    candidate = strip_code_fence(text)
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None, "" if isinstance(parsed, dict) else "json_not_object"
    except Exception as direct_error:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(candidate[start:end + 1])
                return parsed if isinstance(parsed, dict) else None, "" if isinstance(parsed, dict) else "json_not_object"
            except Exception as slice_error:
                recovered = recover_alignment_payload(candidate[start:end + 1])
                if recovered is not None:
                    return recovered, ""
                return None, f"json_parse_error: {slice_error}; direct_error: {direct_error}"
        recovered = recover_alignment_payload(candidate)
        if recovered is not None:
            return recovered, ""
        return None, f"json_parse_error: {direct_error}"


def extract_response_text(response: Any) -> str:
    try:
        return response.choices[0].message.content
    except Exception:
        return str(response)


def make_request(system_prompt: str, user_prompt: str) -> Any:
    from swift.infer_engine import InferRequest

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return InferRequest(messages=messages)


def build_swift_infer(args: argparse.Namespace) -> Any:
    from swift.arguments import InferArguments
    from swift.pipelines.infer import SwiftInfer

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
    if args.vllm_gpu_memory_utilization is not None:
        infer_kwargs["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.vllm_max_model_len is not None:
        infer_kwargs["vllm_max_model_len"] = args.vllm_max_model_len

    infer_args = InferArguments(**infer_kwargs)
    logging.info(
        "ms-swift config | model=%s | backend=%s | template=%s | dtype=%s",
        infer_args.model,
        infer_args.infer_backend,
        infer_args.template,
        infer_args.torch_dtype,
    )
    return SwiftInfer(infer_args)


def make_request_config(args: argparse.Namespace) -> Any:
    from swift.infer_engine import RequestConfig

    return RequestConfig(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        stream=False,
    )
