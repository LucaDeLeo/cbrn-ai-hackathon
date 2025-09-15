# RobustCBRN Eval — Lambda A100 Runbook (20% × 3 datasets)

This runbook gets you from an SSH session on a Lambda A100 instance to validated results and figures under a 3‑hour window. It uses a 20% random sample per dataset, one local HF model, and one seed for speed.

## Prerequisites
- Lambda Cloud A100 (40GB or 80GB) Ubuntu 22.04 (or equivalent CUDA‑capable host)
- Disk: 200–400 GB free on a fast volume for HF caches and logs
- Git, curl, Python 3.10+
- SSH access to the instance

## 0) SSH in and basic checks
```bash
# On your machine
ssh <user>@<lambda-instance-ip>

# On the instance
nvidia-smi               # should show the A100 GPU
which python3 || sudo apt-get update && sudo apt-get install -y python3 git curl
```

## 1) Clone and set up the environment
```bash
git clone https://github.com/apart-research/robustcbrn-eval.git
cd robustcbrn-eval

# Install uv (fast Python toolchain)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps (verifies PyTorch + CUDA)
make setup
```

If `cuda_available=False`, reinstall CUDA wheels inside the venv:
```bash
. .venv/bin/activate
pip uninstall -y torch
pip install --index-url https://download.pytorch.org/whl/cu121 torch
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())
PY
```

## 2) Configure runtime and caches
Create a `.env` tuned for the 3‑hour budget. Example:
```bash
cat > .env <<'ENV'
# One local HF model, one seed (faster)
MODELS=mistralai/Mistral-7B-Instruct-v0.3
SEEDS=123

# Runtime
DEVICE=cuda
DTYPE=bfloat16
CLOZE_MODE=fallback    # faster than HF log-prob; set to 'hf' if you have headroom
LOGS_DIR=logs
RESULTS_DIR=artifacts/results

# Hugging Face cache (point to a large, fast disk)
# HF_HOME=/mnt/ssd/.cache/huggingface
# TRANSFORMERS_CACHE=/mnt/ssd/.cache/huggingface/hub

# Offline after warm (optional)
# HF_HUB_OFFLINE=1
# TRANSFORMERS_OFFLINE=1

# Aggregation (consensus threshold for choices-only exploitable)
CONSENSUS_K=2
ENV
```

Warm the model cache (optional but recommended to avoid first‑run stalls):
```bash
make download-models
```

If a model requires auth, export your token first:
```bash
export HF_TOKEN=hf_xxx
make download-models
```

## 3) Fetch and convert datasets (MMLU‑Pro, HarmBench, WMDP‑Chem)
The registry is pre‑wired with adapters and URLs:
- MMLU‑Pro (Parquet): TIGER‑Lab/MMLU‑Pro test shard
- HarmBench (JSONL): prompts mapped to Refuse/Comply
- WMDP‑Chem (Parquet): chemistry subset

Run:
```bash
make data DATASET=mmlu_pro     # -> data/processed/mmlu_pro/eval.jsonl
make data DATASET=harmbench    # -> data/processed/harmbench/eval.jsonl
make data DATASET=wmdp_chem    # -> data/processed/wmdp_chem/eval.jsonl
```

Validate schemas (optional, recommended):
```bash
.venv/bin/python -m robustcbrn.cli.validate_dataset --schema both data/processed/mmlu_pro/eval.jsonl
.venv/bin/python -m robustcbrn.cli.validate_dataset --schema both data/processed/harmbench/eval.jsonl
.venv/bin/python -m robustcbrn.cli.validate_dataset --schema both data/processed/wmdp_chem/eval.jsonl
```

## 4) Run 20% random samples (1 model × 1 seed)
This loop sizes a 20% subset per dataset, runs MCQ full + choices‑only + Cloze (structured fallback), and aggregates into per‑dataset results directories.
```bash
export DEVICE=cuda DTYPE=bfloat16 CLOZE_MODE=fallback
export MODELS="mistralai/Mistral-7B-Instruct-v0.3" SEEDS=123 CONSENSUS_K=2

for NAME in mmlu_pro harmbench wmdp_chem; do
  DS="data/processed/$NAME/eval.jsonl"
  [ -f "$DS" ] || make data DATASET=$NAME
  N=$(wc -l < "$DS"); SUBSET=$(( (N*20 + 99)/100 ))   # ceil(20%)
  LOGS_DIR="logs/$NAME" RESULTS_DIR="artifacts/results/$NAME"
  echo "Running $NAME on $SUBSET items..."
  DATASET="$DS" SUBSET="$SUBSET" LOGS_DIR="$LOGS_DIR" RESULTS_DIR="$RESULTS_DIR" \
    bash scripts/run_evalset.sh
done
```

Notes:
- For true Cloze log‑prob (more accurate, slower), set `CLOZE_MODE=hf`.
- If you hit OOM, try a smaller model, reduce `SUBSET`, or switch `DTYPE=float16`.

## 5) Inspect results and figures
Per dataset after the loop:
- Results: `artifacts/results/<dataset>/summary.json`, `all_results.csv`
- Figures: aggregator writes to `figs/` (e.g., `mcq_cloze_delta.png`, calibration if present)

Quick checks:
```bash
for NAME in mmlu_pro harmbench wmdp_chem; do
  echo "=== $NAME ==="; sed -n '1,200p' artifacts/results/$NAME/summary.json | head -n 80
done
```

Generate the full figure set (optional):
```bash
make pipeline-figures
```

## 6) Fill the sprint report and validate public artifacts
```bash
make fill-report                            # populates docs/results/report.md from artifacts
bash scripts/validate_release.sh            # enforces no raw text or per-item exploit flags
```

## 7) Copy artifacts off the instance
```bash
tar czf results_bundle.tgz artifacts/results figs docs/results
scp results_bundle.tgz <your-user>@<your-host>:/path/to/save/
```

## Troubleshooting
- PyTorch shows `cuda_available=False`:
  - Reinstall torch with CUDA as in step 1, ensure `nvidia-smi` is present.
- HF 401/403 on download:
  - `export HF_TOKEN=...` then re‑run `make data ...` or `make download-models`.
- OOM / slow runs:
  - Use one 7B–8B model, keep `SEEDS=123`, reduce `SUBSET`, set `DTYPE=float16`.
- Inspect providers not found:
  - The default uses local HF via the Inspect `huggingface/` provider path. Our scripts normalize names, but you can force local HF by leaving `INSPECT_EVAL_MODEL` unset.

## Time/Budget heuristics
- Workload ≈ models × seeds × items. With one 7B model + one seed:
  - A100 bf16 throughput ≈ 25 items/sec → ~0.000011 hours per item per model‑seed.
  - Three datasets × ~20% each typically lands within a ~3‑hour window.

## Optional: Unified pipeline
You can also drive the orchestrator (includes validation, report, etc.):
```bash
bash scripts/run_pipeline.sh --steps validate,setup,sample
# or full suite (heavier):
# bash scripts/run_pipeline.sh --steps validate,setup,discover,sample,full,aggregate,figures,tests,report,verify
```

## Safety & release checklist
Before sharing any outputs:
- Run `bash scripts/validate_release.sh` and review `docs/safety/release-checklist.md`.
- Public artifacts must not include raw question/choices text or per‑item exploit labels.

---

If you need a variant of this runbook for different datasets/models or to use Cloze HF log‑prob by default, copy this file and adjust `MODELS`, `SEEDS`, `CLOZE_MODE`, and the dataset loop.

