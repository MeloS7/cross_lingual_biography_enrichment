#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENRICH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

module load conda
conda activate claim_eval

python "${ENRICH_DIR}/verify_target_en_claim_alignment.py" \
  --model Qwen/Qwen3.5-9B \
  --use-hf \
  --infer-backend vllm \
  --torch-dtype bfloat16 \
  "$@"
