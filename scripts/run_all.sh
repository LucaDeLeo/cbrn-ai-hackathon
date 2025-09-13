#!/usr/bin/env bash
set -euo pipefail

# Edit the model list as needed
MODELS=("meta-llama/Llama-3-8B-Instruct" "mistralai/Mistral-7B-v0.3" "Qwen/Qwen2.5-7B-Instruct")

# 1) MC baseline
for M in "${MODELS[@]}"; do
  lm_eval --model vllm \
    --model_args pretrained=$M,dtype=auto,gpu_memory_utilization=0.90,tensor_parallel_size=1 \
    --tasks tasks/bio_mc.yaml \
    --batch_size auto --log_samples --use_cache cache \
    --output_path results/${M##*/}/mc
done

# 2) Choices-only (for consensus)
for M in "${MODELS[@]}"; do
  lm_eval --model vllm \
    --model_args pretrained=$M,dtype=auto,gpu_memory_utilization=0.90,tensor_parallel_size=1 \
    --tasks tasks/bio_choicesonly.yaml \
    --batch_size auto --log_samples --use_cache cache \
    --output_path results/${M##*/}/choicesonly
done

# 3) Verified cloze
for M in "${MODELS[@]}"; do
  lm_eval --model vllm \
    --model_args pretrained=$M,dtype=auto,gpu_memory_utilization=0.90,tensor_parallel_size=1 \
    --tasks tasks/bio_cloze.yaml \
    --batch_size auto --log_samples --use_cache cache \
    --output_path results/${M##*/}/cloze
done

# 4) Post-processing
python scripts/vote_consensus.py --glob "results/*/choicesonly/*/samples.jsonl" --out results/consensus_choicesonly.csv
python scripts/heuristics.py data/toy_bio_eval.jsonl
# For abstention, point to one cloze run's samples.jsonl:
# python scripts/abstain_ci.py --samples results/<model>/cloze/<run-id>/samples.jsonl

