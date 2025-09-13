#!/usr/bin/env bash
set -euo pipefail

# Orchestrator for robustness checks (module 4: benign pairs)
# - Runs the benign policy-compliance pairs task across models/seeds
# - Aggregates logs into artifacts/results

if [ -f .env ]; then
  set -a; source .env; set +a
fi

BENIGN_DATASET=${BENIGN_DATASET:-data/benign_pairs_sanitized.jsonl}
LOGS_DIR=${LOGS_DIR:-logs}
RESULTS_DIR=${RESULTS_DIR:-artifacts/results}
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"

IFS=';' read -ra MODELS_ARR <<< "${MODELS:-meta-llama/Llama-3.1-8B-Instruct; mistralai/Mistral-7B-Instruct-v0.3}"
IFS=';' read -ra SEEDS_ARR <<< "${SEEDS:-123;456}"

SUBSET=${SUBSET:-128}

echo "[robustness_suite] Benign pairs on $BENIGN_DATASET"
for M in "${MODELS_ARR[@]}"; do
  for S in "${SEEDS_ARR[@]}"; do
    echo "[robustness_suite] benign_policy_pairs model=$M seed=$S"
    .venv/bin/inspect eval robustcbrn.tasks.benign_policy_pairs:benign_policy_pairs \
      --arg dataset_path="$BENIGN_DATASET" \
      --arg seed="$S" \
      --arg max_items="$SUBSET" \
      --model "$M" \
      --log-dir "$LOGS_DIR"
  done
done

echo "[robustness_suite] Aggregating"
.venv/bin/python -m robustcbrn.analysis.aggregate --logs "$LOGS_DIR" --out "$RESULTS_DIR"

echo "[robustness_suite] Done."

