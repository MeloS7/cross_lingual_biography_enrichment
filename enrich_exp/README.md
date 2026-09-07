# Biography Enrichment Experiments

This directory provides the code for selecting target-language claims for English-biography enrichment and evaluating claims before and after enrichment.

Run all commands below from the repository root.

## Input files

The claim-selection and before/after matching steps use:

```text
datasets/
├── pool_300_en_claims_gpt_5_1.jsonl
└── pool_300_target_claims_gpt_5_1.jsonl
```

Post-enrichment claims and existing similarity candidates are organized as:

```text
datasets/claim_enrich/
└── <method>/<target_lang>/<model_name>/
    ├── after_enrich_claims.jsonl
    └── before_after_enrich_similarity.jsonl
```

Available values are:

- `method`: `claim_enrich`, `target_enrich`, or `translate_enrich`
- `target_lang`: `az`, `fr`, or `zh`
- `model_name`: `gemma4`, `mistral`, or `qwen`

## 1. Retrieve target-to-English claim candidates

```bash
python enrich_exp/process/retrieve_target_en_claim_candidates.py \
  --device cuda
```

For CPU execution, use `--device cpu`.

## 2. Verify target-to-English claim relations

The provided launcher uses `Qwen/Qwen3.5-9B` with the vLLM backend:

```bash
bash enrich_exp/scripts/verify_target_en_claim.sh
```

## 3. Select claims for enrichment

```bash
python enrich_exp/process/select_enrichment_claims.py
```

## 4. Match claims before and after enrichment

Run candidate retrieval for one method, language, and model combination:

```bash
python enrich_exp/process/retrieve_enrich_result_candidates.py \
  --method claim_enrich \
  --target-lang fr \
  --model-name-dir gemma4 \
  --device cuda
```

Change `--method`, `--target-lang`, and `--model-name-dir` to run another combination.

## 5. Verify before/after claim relations

Set the method, language, and model directory, then run:

```bash
METHOD=claim_enrich \
TARGET_LANG=fr \
MODEL_NAME=gemma4 \
  bash enrich_exp/scripts/verify_before_after_enrich.sh
```

The launcher uses `Qwen/Qwen3.5-9B` with the vLLM backend.

## 6. Compute enrichment metrics

After all required verification runs are complete, aggregate the metrics with:

```bash
python enrich_exp/process/compute_before_after_enrich_metrics.py
```
