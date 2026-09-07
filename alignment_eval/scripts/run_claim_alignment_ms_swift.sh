#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EVAL_DIR}/.." && pwd)"
PIPELINE_PY="${EVAL_DIR}/claim_alignment_ms_swift_pipeline.py"
MODEL_ID="Qwen/Qwen3.5-9B"
MODE="vllm"
PROMPT_TEMPLATE="claim_only"
EXTRA_ARGS=("$@")

case "${PROMPT_TEMPLATE}" in
  claim_only)
    PROMPTS_YAML="${REPO_DIR}/prompts/claim_alignment_prompts_claim_only.yaml"
    ;;
  claim_infobox)
    PROMPTS_YAML="${REPO_DIR}/prompts/claim_alignment_prompts_claim_infobox.yaml"
    ;;
  *)
    echo "Invalid prompt template: ${PROMPT_TEMPLATE}"
    echo "Valid values: claim_only | claim_infobox"
    exit 1
    ;;
esac

MODEL_NAME="$(basename "${MODEL_ID}")"
RESULTS_DIR="${REPO_DIR}/results/alignment_eval/${MODEL_NAME}/${PROMPT_TEMPLATE}"
mkdir -p "${RESULTS_DIR}"

case "${MODE}" in
  vllm)
    python "${PIPELINE_PY}" \
      --model "${MODEL_ID}" \
      --use-hf \
      --infer-backend vllm \
      --vllm-max-model-len 16384 \
      --vllm-gpu-memory-utilization 0.95 \
      --temperature 0 \
      --top-p 1 \
      --max-new-tokens 8192 \
      --results-dir "${RESULTS_DIR}" \
      --prompts-yaml "${PROMPTS_YAML}" \
      --output-jsonl "claim_alignment_predictions.jsonl" \
      --metrics-path "claim_alignment_metrics.json" \
      --log-file "run.log" \
      "${EXTRA_ARGS[@]}"
    ;;
  transformers)
    python "${PIPELINE_PY}" \
      --model "${MODEL_ID}" \
      --use-hf \
      --infer-backend transformers \
      --max-batch-size 16 \
      --temperature 0 \
      --top-p 1 \
      --max-new-tokens 8192 \
      --results-dir "${RESULTS_DIR}" \
      --prompts-yaml "${PROMPTS_YAML}" \
      --output-jsonl "claim_alignment_predictions.jsonl" \
      --metrics-path "claim_alignment_metrics.json" \
      --log-file "run.log" \
      "${EXTRA_ARGS[@]}"
    ;;
  *)
    echo "Invalid mode: ${MODE}"
    echo "Valid values: vllm | transformers"
    exit 1
    ;;
esac
