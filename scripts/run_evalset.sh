#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then
  set -a; source .env; set +a
fi

DATASET=${DATASET:-data/sample_sanitized.jsonl}
LOGS_DIR=${LOGS_DIR:-logs}
RESULTS_DIR=${RESULTS_DIR:-artifacts/results}
CONSENSUS_K=${CONSENSUS_K:-2}
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

IFS=';' read -ra MODELS_ARR <<< "${MODELS:-meta-llama/Llama-3.1-8B-Instruct; mistralai/Mistral-7B-Instruct-v0.3}"
IFS=';' read -ra SEEDS_ARR <<< "${SEEDS:-123;456}"

SUBSET=${SUBSET:-128}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}
CLOZE_MODE=${CLOZE_MODE:-fallback}

MODELS_COUNT=${#MODELS_ARR[@]}
SEEDS_COUNT=${#SEEDS_ARR[@]}

# Workload summary (no budget guard)
echo "[run_evalset] Workload: models×seeds×items = ${MODELS_COUNT}×${SEEDS_COUNT}×${SUBSET} = $((MODELS_COUNT*SEEDS_COUNT*SUBSET))"

# Timing start
run_start=$(date +%s)

# Pre-eval schema validation (MCQ and choices-only)
echo "[run_evalset] Validating dataset schema for MCQ and choices-only"
if ! .venv/bin/python -m robustcbrn.cli.validate_dataset --schema both "$DATASET"; then
  echo "[run_evalset] Dataset schema invalid. See docs/getting-started/usage.md#dataset-schema"
  exit 4
fi

for M in "${MODELS_ARR[@]}"; do
  for S in "${SEEDS_ARR[@]}"; do
    # Normalize model name for Inspect providers (prefix huggingface/ if needed)
    IM="$M"
    case "$M" in
      openai/*|anthropic/*|google/*|groq/*|mistral/*|togetherai/*|huggingface/*|hf/*|vllm/*)
        IM="$M" ;;
      *)
        IM="huggingface/$M" ;;
    esac
    echo "[run_evalset] mcq_full model=$M seed=$S"
    if ! .venv/bin/inspect eval robustcbrn.tasks.mcq_full:mcq_full -T dataset_path="$DATASET" -T seed="$S" -T max_items="$SUBSET" --model "$IM" --log-dir "$LOGS_DIR"; then
      echo "[run_evalset] WARN: mcq_full failed (provider not available?); continuing"
    fi
    echo "[run_evalset] mcq_choices_only model=$M seed=$S"
    if ! .venv/bin/inspect eval robustcbrn.tasks.mcq_choices_only:mcq_choices_only -T dataset_path="$DATASET" -T seed="$S" -T max_items="$SUBSET" --model "$IM" --log-dir "$LOGS_DIR"; then
      echo "[run_evalset] WARN: mcq_choices_only failed (provider not available?); continuing"
    fi
    if [ "$CLOZE_MODE" = "hf" ]; then
      echo "[run_evalset] cloze_hf_logprob model=$M seed=$S"
      .venv/bin/python -m robustcbrn.tasks.cloze_logprob --dataset_path "$DATASET" --seed "$S" --max_items "$SUBSET" --model "$M" --device "$DEVICE" --dtype "$DTYPE" --logs_dir "$LOGS_DIR"
    else
      echo "[run_evalset] cloze_full (fallback structured) model=$M seed=$S"
      if ! .venv/bin/inspect eval robustcbrn.tasks.cloze_full:cloze_full -T dataset_path="$DATASET" -T seed="$S" -T max_items="$SUBSET" --model "$IM" --log-dir "$LOGS_DIR"; then
        echo "[run_evalset] WARN: cloze_full fallback failed (provider not available?); continuing"
      fi
    fi
  done
done

echo "[run_evalset] Aggregating (k=$CONSENSUS_K)"
.venv/bin/python -m robustcbrn.analysis.aggregate --logs "$LOGS_DIR" --out "$RESULTS_DIR" --k "$CONSENSUS_K"

# Summary timing
run_end=$(date +%s)
elapsed=$((run_end - run_start))
echo "[run_evalset] Summary: elapsed=${elapsed}s models=${MODELS_COUNT} seeds=${SEEDS_COUNT} subset=${SUBSET} k=${CONSENSUS_K}"
echo "[run_evalset] Done."
