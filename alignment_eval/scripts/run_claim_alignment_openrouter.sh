#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EVAL_DIR}/.." && pwd)"
PIPELINE_PY="${EVAL_DIR}/claim_alignment_openrouter_pipeline.py"
MODEL_ID="openai/gpt-5.1"
PROMPT_TEMPLATE="claim_infobox"
API_KEY="${OPENROUTER_API_KEY:-}"
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

if [[ -z "${API_KEY}" ]]; then
  echo "ERROR: OpenRouter API key is empty."
  echo "Please export OPENROUTER_API_KEY first."
  exit 1
fi

MODEL_NAME="$(basename "${MODEL_ID}")"
RESULTS_DIR="${REPO_DIR}/results/alignment_eval/${MODEL_NAME}/${PROMPT_TEMPLATE}"
mkdir -p "${RESULTS_DIR}"

python "${PIPELINE_PY}" \
  --model "${MODEL_ID}" \
  --api-key "${API_KEY}" \
  --temperature 0 \
  --top-p 1 \
  --max-new-tokens 8192 \
  --results-dir "${RESULTS_DIR}" \
  --prompts-yaml "${PROMPTS_YAML}" \
  --output-jsonl "claim_alignment_predictions.jsonl" \
  --metrics-path "claim_alignment_metrics.json" \
  --cost-log-jsonl "claim_alignment_cost.jsonl" \
  "${EXTRA_ARGS[@]}"
