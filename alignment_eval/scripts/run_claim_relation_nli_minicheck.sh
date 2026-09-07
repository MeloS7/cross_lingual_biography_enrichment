#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd "${EVAL_DIR}/.." && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python "${EVAL_DIR}/claim_relation_nli_minicheck.py" \
  --input-path "${REPO_DIR}/datasets/CLAW_4L_RC.jsonl" \
  --results-dir "${REPO_DIR}/results/alignment_eval/claim_relation_nli_minicheck" \
  --deberta-model "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli" \
  --minicheck-model "Bespoke-MiniCheck-7B" \
  --minicheck-cache-dir "${EVAL_DIR}/ckpts" \
  --batch-size 32 \
  "$@"
