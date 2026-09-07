#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENRICH_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${ENRICH_DIR}/.." && pwd)"
DATASET_DIR="${PROJECT_DIR}/datasets/claim_enrich"
PIPELINE_PY="${ENRICH_DIR}/process/retrieve_enrich_result_candidates.py"

BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-}"
LIMIT="${LIMIT:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
DETAILS_JSONL="${DETAILS_JSONL:-before_after_enrich_similarity_all.jsonl}"

module load conda
conda activate claim_eval

mapfile -t AFTER_FILES < <(find "${DATASET_DIR}" -mindepth 4 -maxdepth 4 -type f -name "after_enrich_claims.jsonl" | sort)

if [[ "${#AFTER_FILES[@]}" -eq 0 ]]; then
  echo "No after_enrich_claims.jsonl files found under ${DATASET_DIR}" >&2
  exit 1
fi

echo "Found ${#AFTER_FILES[@]} after-enrichment claim files."
echo "mode=all_similarities output=${DETAILS_JSONL}"

for after_jsonl in "${AFTER_FILES[@]}"; do
  rel_path="${after_jsonl#${DATASET_DIR}/}"
  method="$(cut -d/ -f1 <<< "${rel_path}")"
  target_lang="$(cut -d/ -f2 <<< "${rel_path}")"
  model_name_dir="$(cut -d/ -f3 <<< "${rel_path}")"
  output_jsonl="$(dirname "${after_jsonl}")/${DETAILS_JSONL}"

  if [[ "${SKIP_EXISTING}" == "1" && -s "${output_jsonl}" ]]; then
    echo "Skipping existing: ${method}/${target_lang}/${model_name_dir}/${DETAILS_JSONL}"
    continue
  fi

  echo "Running: method=${method} target_lang=${target_lang} model=${model_name_dir}"

  cmd=(
    python "${PIPELINE_PY}"
    --method "${method}"
    --target-lang "${target_lang}"
    --model-name-dir "${model_name_dir}"
    --all-similarities
    --batch-size "${BATCH_SIZE}"
    --details-jsonl "${DETAILS_JSONL}"
  )

  if [[ -n "${DEVICE}" ]]; then
    cmd+=(--device "${DEVICE}")
  fi
  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
    cmd+=(--local-files-only)
  fi

  "${cmd[@]}"
done

echo "Finished all retrieval jobs."
