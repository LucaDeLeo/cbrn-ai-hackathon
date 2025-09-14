#!/usr/bin/env bash
set -euo pipefail

# Load optional .env
if [ -f .env ]; then
  set -a; source .env; set +a
fi

DATASET="data/sample_sanitized.jsonl"
LOGS_DIR=${LOGS_DIR:-logs}
RESULTS_DIR=${RESULTS_DIR:-artifacts/results}
CONSENSUS_K=${CONSENSUS_K:-2}
mkdir -p "$LOGS_DIR"

MODEL=${MODEL:-${INSPECT_EVAL_MODEL:-}} # prefer API model if explicitly set
if [ -z "${MODEL}" ]; then
  # default to first local model name
  IFS=';' read -r first_model _ <<< "${MODELS:-meta-llama/Llama-3.1-8B-Instruct}"
  MODEL="${first_model}"
fi

echo "[run_sample] Using model: $MODEL"



echo "[run_sample] Running mcq_full"
.venv/bin/inspect eval robustcbrn.tasks.mcq_full:mcq_full -T dataset_path="$DATASET" --model "$MODEL" --log-dir "$LOGS_DIR"

echo "[run_sample] Running mcq_choices_only"
.venv/bin/inspect eval robustcbrn.tasks.mcq_choices_only:mcq_choices_only -T dataset_path="$DATASET" --model "$MODEL" --log-dir "$LOGS_DIR"

echo "[run_sample] Running cloze_full (fallback structured)"
.venv/bin/inspect eval robustcbrn.tasks.cloze_full:cloze_full -T dataset_path="$DATASET" --model "$MODEL" --log-dir "$LOGS_DIR"

echo "[run_sample] Aggregating (k=$CONSENSUS_K)"
.venv/bin/python -m robustcbrn.analysis.aggregate --logs "$LOGS_DIR" --out "$RESULTS_DIR" --k "$CONSENSUS_K"

echo "[run_sample] Done. Results under artifacts/results."
