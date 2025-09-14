#!/usr/bin/env bash
set -euo pipefail

# One-shot orchestrator for full suite:
# - MCQ full, MCQ choices-only, Cloze (per CLOZE_MODE)
# - Benign pairs, Paraphrase consistency, Perturbation stability
# - Aggregation with consensus threshold k (CONSENSUS_K)

if [ -f .env ]; then
  set -a; source .env; set +a
fi

DATASET=${DATASET:-data/sample_sanitized.jsonl}
BENIGN_DATASET=${BENIGN_DATASET:-data/benign_pairs_sanitized.jsonl}
LOGS_DIR=${LOGS_DIR:-logs}
RESULTS_DIR=${RESULTS_DIR:-artifacts/results}
CONSENSUS_K=${CONSENSUS_K:-2}

mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

IFS=';' read -ra MODELS_ARR <<< "${MODELS:-meta-llama/Llama-3.1-8B-Instruct; mistralai/Mistral-7B-Instruct-v0.3}"
IFS=';' read -ra SEEDS_ARR <<< "${SEEDS:-123;456}"
SUBSET=${SUBSET:-128}
CLOZE_MODE=${CLOZE_MODE:-fallback}
DEVICE=${DEVICE:-cuda}
DTYPE=${DTYPE:-bfloat16}

echo "[run_all] Starting full suite (k=$CONSENSUS_K)"
all_start=$(date +%s)

# Phase 1: MCQ + Cloze via existing orchestrator
echo "[run_all] Phase 1: Evalset"
b1=$(date +%s)
bash scripts/run_evalset.sh
b1_end=$(date +%s)
echo "[run_all] Phase 1 elapsed=$((b1_end - b1))s"

# Prepare Inspect provider model prefix normalization
normalize_model() {
  local m="$1"
  case "$m" in
    openai/*|anthropic/*|google/*|groq/*|mistral/*|togetherai/*|huggingface/*|hf/*|vllm/*)
      echo "$m" ;;
    *)
      echo "huggingface/$m" ;;
  esac
}

# Phase 2: Benign pairs
echo "[run_all] Phase 2: Benign pairs on $BENIGN_DATASET"
b2=$(date +%s)
for M in "${MODELS_ARR[@]}"; do
  for S in "${SEEDS_ARR[@]}"; do
    IM=$(normalize_model "$M")
    echo "[run_all] benign_policy_pairs model=$M seed=$S"
    if ! .venv/bin/inspect eval robustcbrn.tasks.benign_policy_pairs:benign_policy_pairs \
      -T dataset_path="$BENIGN_DATASET" \
      -T seed="$S" \
      -T max_items="$SUBSET" \
      --model "$IM" \
      --log-dir "$LOGS_DIR"; then
      echo "[run_all] WARN: benign_policy_pairs failed; continuing"
    fi
  done
done
b2_end=$(date +%s)
echo "[run_all] Phase 2 elapsed=$((b2_end - b2))s"

# Phase 3: Paraphrase consistency (dataset-based)
echo "[run_all] Phase 3: Paraphrase consistency on $DATASET"
b3=$(date +%s)
for M in "${MODELS_ARR[@]}"; do
  for S in "${SEEDS_ARR[@]}"; do
    IM=$(normalize_model "$M")
    echo "[run_all] paraphrase_consistency model=$M seed=$S"
    if ! .venv/bin/inspect eval robustcbrn.tasks.paraphrase_consistency:paraphrase_consistency \
      -T dataset_path="$DATASET" \
      -T seed="$S" \
      -T max_items="$SUBSET" \
      --model "$IM" \
      --log-dir "$LOGS_DIR"; then
      echo "[run_all] WARN: paraphrase_consistency failed; continuing"
    fi
  done
done
b3_end=$(date +%s)
echo "[run_all] Phase 3 elapsed=$((b3_end - b3))s"

# Phase 4: Perturbation stability (dataset-based)
echo "[run_all] Phase 4: Perturbation stability on $DATASET"
b4=$(date +%s)
for M in "${MODELS_ARR[@]}"; do
  for S in "${SEEDS_ARR[@]}"; do
    IM=$(normalize_model "$M")
    echo "[run_all] perturbation_stability model=$M seed=$S"
    if ! .venv/bin/inspect eval robustcbrn.tasks.perturbation_stability:perturbation_stability \
      -T dataset_path="$DATASET" \
      -T seed="$S" \
      -T max_items="$SUBSET" \
      --model "$IM" \
      --log-dir "$LOGS_DIR"; then
      echo "[run_all] WARN: perturbation_stability failed; continuing"
    fi
  done
done
b4_end=$(date +%s)
echo "[run_all] Phase 4 elapsed=$((b4_end - b4))s"

# Final aggregation
echo "[run_all] Aggregating (k=$CONSENSUS_K)"
bagg=$(date +%s)
.venv/bin/python -m robustcbrn.analysis.aggregate --logs "$LOGS_DIR" --out "$RESULTS_DIR" --k "$CONSENSUS_K"
bagg_end=$(date +%s)
echo "[run_all] Aggregation elapsed=$((bagg_end - bagg))s"

all_end=$(date +%s)
elapsed=$((all_end - all_start))
echo "[run_all] Summary: elapsed=${elapsed}s models=${#MODELS_ARR[@]} seeds=${#SEEDS_ARR[@]} subset=${SUBSET} k=${CONSENSUS_K}"
echo "[run_all] Done."
