#!/usr/bin/env bash
set -euo pipefail

# Requires: uv (https://docs.astral.sh/uv/)

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] ERROR: 'uv' is not installed. Install from https://docs.astral.sh/uv/getting-started/"
  echo "        e.g., curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 2
fi

PY_VERSION=${PY_VERSION:-3.10}
echo "[setup] Creating virtual environment (.venv) with uv (Python ${PY_VERSION})"
uv venv .venv --python "${PY_VERSION}"

echo "[setup] Installing requirements with uv"
uv pip install --python .venv -r requirements.txt

echo "[setup] Verifying Torch + CUDA"
.venv/bin/python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
PY

echo "[setup] Locking requirements to requirements.lock.txt"
uv pip freeze --python .venv > requirements.lock.txt

echo "[setup] Writing .env.example"
cat > .env.example <<'ENV'
# Cloud budget guard
CLOUD_BUDGET_USD=400
# Set your GPU hourly rate (e.g., 1x A100 ~ $1.10–$2.10/hr depending on provider)
GPU_HOURLY_USD=
# Optional budget to cap API spend
API_BUDGET_USD=0

# Default local HF models (semicolon-separated)
MODELS=meta-llama/Llama-3.1-8B-Instruct; mistralai/Mistral-7B-Instruct-v0.3; Qwen/Qwen2.5-7B-Instruct

# Optional API model for Inspect providers (off by default)
INSPECT_EVAL_MODEL=

# Runtime
DEVICE=cuda
DTYPE=bfloat16
BATCH_SIZE=4
MAX_SEQ_LEN=4096
SEEDS=123;456

# Paths
LOGS_DIR=logs
RESULTS_DIR=artifacts/results
FIGS_DIR=artifacts/figs
BUDGET_DIR=.budget
ENV

echo "[setup] Done. Use .venv/bin/python, or 'uv run --python .venv <cmd>'"
