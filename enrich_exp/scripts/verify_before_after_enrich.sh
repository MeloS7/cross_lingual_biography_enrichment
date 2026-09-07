#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENRICH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${ENRICH_DIR}/.." && pwd)"

METHOD="${METHOD:-target_enrich}"
MODEL_NAME="${MODEL_NAME:-gemma4}"
TARGET_LANG="${TARGET_LANG:-fr}"

INPUT_JSONL="${INPUT_JSONL:-${PROJECT_DIR}/datasets/claim_enrich/${METHOD}/${TARGET_LANG}/${MODEL_NAME}/before_after_enrich_similarity.jsonl}"
RESULTS_DIR="${RESULTS_DIR:-${PROJECT_DIR}/results/enrich_exp/before_after_enrich_verify/${METHOD}/${TARGET_LANG}/${MODEL_NAME}}"

module load conda
conda activate claim_eval

python "${ENRICH_DIR}/verify_before_after_enrich.py" \
  --input-jsonl "${INPUT_JSONL}" \
  --results-dir "${RESULTS_DIR}" \
  --model Qwen/Qwen3.5-9B \
  --use-hf \
  --infer-backend vllm \
  --torch-dtype bfloat16 \
  "$@"
