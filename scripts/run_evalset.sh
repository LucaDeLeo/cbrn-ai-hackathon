#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then
  set -a; source .env; set +a
fi

DATASET=${DATASET:-data/sample_sanitized.jsonl}
LOGS_DIR=${LOGS_DIR:-logs}
mkdir -p "$LOGS_DIR"

IFS=';' read -ra MODELS_ARR <<< "${MODELS:-meta-llama/Llama-3.1-8B-Instruct; mistralai/Mistral-7B-Instruct-v0.3}"
IFS=';' read -ra SEEDS_ARR <<< "${SEEDS:-123;456}"

SUBSET=${SUBSET:-128}

HOURLY=${GPU_HOURLY_USD:-0}
MODELS_COUNT=${#MODELS_ARR[@]}
SEEDS_COUNT=${#SEEDS_ARR[@]}

PROJECTED_HOURS=$(.venv/bin/python - <<PY
models=${MODELS_COUNT}
seeds=${SEEDS_COUNT}
items=${SUBSET}
# crude estimate: 25 it/s @ 8B, 0.000011h/item/model (~0.04s)
hours = models*seeds*items*0.000011
print(f"{hours:.4f}")
PY
)

echo "[run_evalset] Projected hours: $PROJECTED_HOURS"
.venv/bin/python -m robustcbrn.budget_guard "evalset" --dry-run --projected-hours "$PROJECTED_HOURS" --hourly-usd "$HOURLY"

for M in "${MODELS_ARR[@]}"; do
  for S in "${SEEDS_ARR[@]}"; do
    echo "[run_evalset] mcq_full model=$M seed=$S"
    .venv/bin/inspect eval robustcbrn.tasks.mcq_full:mcq_full --arg dataset_path="$DATASET" --arg seed="$S" --arg max_items="$SUBSET" --model "$M" --log-dir "$LOGS_DIR"
    echo "[run_evalset] mcq_choices_only model=$M seed=$S"
    .venv/bin/inspect eval robustcbrn.tasks.mcq_choices_only:mcq_choices_only --arg dataset_path="$DATASET" --arg seed="$S" --arg max_items="$SUBSET" --model "$M" --log-dir "$LOGS_DIR"
    echo "[run_evalset] cloze_full model=$M seed=$S"
    .venv/bin/inspect eval robustcbrn.tasks.cloze_full:cloze_full --arg dataset_path="$DATASET" --arg seed="$S" --arg max_items="$SUBSET" --model "$M" --log-dir "$LOGS_DIR"
  done
done

echo "[run_evalset] Aggregating"
.venv/bin/python -m robustcbrn.analysis.aggregate --logs "$LOGS_DIR" --out "${RESULTS_DIR:-artifacts/results}"

echo "[run_evalset] Done."
