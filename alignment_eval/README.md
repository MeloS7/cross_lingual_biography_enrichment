# Claim-Pair Alignment Evaluation

This directory provides the code for claim-pair relation classification with embedding similarity, NLI/MiniCheck baselines, local LLMs, and API models.

Run all commands below from the repository root.

## Input files

The evaluation dataset is:

```text
datasets/CLAW_4L_RC.jsonl
```

The LLM pipelines use prompts stored under:

```text
prompts/
├── claim_alignment_prompts_claim_only.yaml
└── claim_alignment_prompts_claim_infobox.yaml
```

## 1. Embedding similarity

Run the BGE and MPNet similarity analysis with:

```bash
python alignment_eval/embedding_claim_similarity.py --device cuda
```

For CPU execution:

```bash
python alignment_eval/embedding_claim_similarity.py --device cpu
```

## 2. NLI and MiniCheck baselines

Run the DeBERTa NLI and Bespoke-MiniCheck baselines with:

```bash
bash alignment_eval/scripts/run_claim_relation_nli_minicheck.sh
```

To select another GPU:

```bash
CUDA_VISIBLE_DEVICES=1 \
  bash alignment_eval/scripts/run_claim_relation_nli_minicheck.sh
```

## 3. Local LLM evaluation

The provided launcher uses `Qwen/Qwen3.5-9B`, the claim-only prompt, and the vLLM backend:

```bash
bash alignment_eval/scripts/run_claim_alignment_ms_swift.sh
```

To run the Python entry point directly:

```bash
python alignment_eval/claim_alignment_ms_swift_pipeline.py \
  --model Qwen/Qwen3.5-9B \
  --use-hf \
  --infer-backend vllm \
  --prompts-yaml prompts/claim_alignment_prompts_claim_only.yaml
```

The supported inference backends are `transformers`, `vllm`, `sglang`, and `lmdeploy`, subject to support for the selected model.

## 4. OpenRouter evaluation

Set the API key and run the OpenRouter launcher with:

```bash
export OPENROUTER_API_KEY="YOUR_API_KEY"
bash alignment_eval/scripts/run_claim_alignment_openrouter.sh
```

The launcher uses `openai/gpt-5.1` and the claim-plus-infobox prompt by default. The model and prompt configuration can be changed at the top of the shell script.

## 5. Build the accuracy table

After the evaluation runs are complete, aggregate their metrics with:

```bash
python alignment_eval/build_accuracy_table.py
```
