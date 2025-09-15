#!/usr/bin/env bash
# Remote orchestrator for RobustCBRN runs on a Lambda GPU instance.
# Supports key-based SSH by default; optionally uses sshpass if SSHPASS is set.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: remote_run_lambda.sh --host <ip> --user <name> --key <path>
       [--models "mistralai/Mistral-7B-Instruct-v0.3"] [--seeds 123]
       [--datasets "mmlu_pro,harmbench,wmdp_chem"] [--cloze-mode fallback]
       [--dtype bfloat16] [--device cuda]

Env support:
  SSHPASS              Password for sshpass (optional; key-based recommended)
  RESULTS_LOCAL_DIR    Where to copy results (default: ./remote_results)

Examples:
  bash scripts/remote_run_lambda.sh --host 1.2.3.4 --user ubuntu --key ~/.ssh/id_rsa
  SSHPASS='mypw' bash scripts/remote_run_lambda.sh --host 1.2.3.4 --user ubuntu --key ''
EOF
}

HOST=""
USER="ubuntu"
KEY=""
MODELS="mistralai/Mistral-7B-Instruct-v0.3"
SEEDS="123"
DATASETS="mmlu_pro,harmbench,wmdp_chem"
CLOZE_MODE="fallback"
DTYPE="bfloat16"
DEVICE="cuda"
RESULTS_LOCAL_DIR="${RESULTS_LOCAL_DIR:-remote_results}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2;;
    --user) USER="$2"; shift 2;;
    --key) KEY="$2"; shift 2;;
    --models) MODELS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --datasets) DATASETS="$2"; shift 2;;
    --cloze-mode) CLOZE_MODE="$2"; shift 2;;
    --dtype) DTYPE="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown arg: $1"; usage; exit 2;;
  esac
done

if [[ -z "$HOST" ]]; then echo "--host is required"; usage; exit 2; fi
if [[ -z "${SSHPASS:-}" ]] && [[ -z "$KEY" ]]; then
  echo "Either set --key for key-based auth or SSHPASS for sshpass."; exit 2
fi

SSH_BASE=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
SCP_BASE=(scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

if [[ -n "${SSHPASS:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass not found on local machine"; exit 3
  fi
  SSH_CMD=(sshpass -p "$SSHPASS" "${SSH_BASE[@]}" "$USER@$HOST")
  SCP_CMD=(sshpass -p "$SSHPASS" "${SCP_BASE[@]}")
else
  SSH_CMD=("${SSH_BASE[@]}" -i "$KEY" "$USER@$HOST")
  SCP_CMD=("${SCP_BASE[@]}" -i "$KEY")
fi

REMOTE_SCRIPT=$(cat <<'"EOS"'
set -euo pipefail
echo "[remote] OS: $(uname -a)"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "[remote] nvidia-smi not found (container?)"

# Basics
sudo apt-get update -y || true
sudo apt-get install -y git curl python3 python3-venv || true

# Clone or update repo
if [[ -d robustcbrn-eval/.git ]]; then
  cd robustcbrn-eval
  git pull --ff-only || true
else
  git clone https://github.com/apart-research/robustcbrn-eval.git
  cd robustcbrn-eval
fi

# Install uv and set up env
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

make setup

# Write .env from injected variables
cat > .env <<ENV
MODELS=${MODELS}
SEEDS=${SEEDS}
DEVICE=${DEVICE}
DTYPE=${DTYPE}
CLOZE_MODE=${CLOZE_MODE}
LOGS_DIR=logs
RESULTS_DIR=artifacts/results
FIGS_DIR=artifacts/figs
CONSENSUS_K=2
ENV

# Optional: warm models
make download-models || true

# Fetch/convert datasets
IFS=',' read -ra DS_ARR <<< "${DATASETS}"
for D in "${DS_ARR[@]}"; do
  make data DATASET="$D" || true
done

# 20% run per dataset
export DEVICE=${DEVICE} DTYPE=${DTYPE} CLOZE_MODE=${CLOZE_MODE}
export MODELS=${MODELS} SEEDS=${SEEDS} CONSENSUS_K=2

for D in "${DS_ARR[@]}"; do
  DS_PATH="data/processed/${D}/eval.jsonl"
  if [[ ! -f "$DS_PATH" ]]; then
    echo "[remote] Missing processed dataset: $D"; continue
  fi
  N=$(wc -l < "$DS_PATH" | tr -d ' ')
  SUBSET=$(( (N*20 + 99)/100 ))
  LOGS_DIR="logs/${D}" RESULTS_DIR="artifacts/results/${D}"
  echo "[remote] Running $D on $SUBSET items"
  DATASET="$DS_PATH" SUBSET="$SUBSET" LOGS_DIR="$LOGS_DIR" RESULTS_DIR="$RESULTS_DIR" \
    bash scripts/run_evalset.sh || true
done

# Bundle results
tar czf results_bundle.tgz artifacts/results figs docs/results || tar czf results_bundle.tgz artifacts/results docs/results || true
echo "[remote] Bundle at: $(pwd)/results_bundle.tgz"
"EOS"
)

echo "[local] Starting remote run on $USER@$HOST"
"${SSH_CMD[@]}" bash -s <<EOF
MODELS="$MODELS" SEEDS="$SEEDS" DATASETS="$DATASETS" CLOZE_MODE="$CLOZE_MODE" DTYPE="$DTYPE" DEVICE="$DEVICE"
$REMOTE_SCRIPT
EOF

mkdir -p "$RESULTS_LOCAL_DIR"
echo "[local] Copying results bundle..."
${SCP_CMD[@]} "$USER@$HOST:robustcbrn-eval/results_bundle.tgz" "$RESULTS_LOCAL_DIR/" || echo "[local] Copy failed; check remote path"
echo "[local] Done. Bundle at $RESULTS_LOCAL_DIR/results_bundle.tgz"

