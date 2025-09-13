# robustcbrn-eval

Robust evaluation pipeline on top of [EleutherAI lm-evaluation-harness], focused on:
- **MC vs choices-only vs verified cloze**
- **Consensus over choices-only** (shortcut detection)
- **Heuristics baselines**
- **Abstention + calibration with CIs**
- **Permutation sensitivity**

> **Safety first:** This repo is evaluation-only. It includes toy data and templates only. Do **not** add sensitive or hazardous content. Logs use hashed IDs and are intended for private analysis.

## Quick start (Lambda A100 or any CUDA host)

```bash
# 1) Create env
conda env create -f environment.yml && conda activate robustcbrn

# 2) Install deps (CUDA/torch assumed installed on Lambda Stack)
pip install -r requirements.txt

# 3) Sanity check GPU
python - <<'PY'
import torch; print("CUDA:", torch.cuda.is_available())
PY

# 4) List tasks
lm_eval --tasks list | sed -n '1,50p'  # sanity

# 5) Run toy tasks (HF backend)
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3-8B-Instruct,dtype=bfloat16 \
  --tasks tasks/bio_mc.yaml \
  --device cuda:0 --batch_size auto \
  --log_samples --use_cache cache \
  --output_path results/llama3-8b/mc

# 6) vLLM (often faster)
lm_eval --model vllm \
  --model_args pretrained=mistralai/Mistral-7B-v0.3,dtype=auto,gpu_memory_utilization=0.90,tensor_parallel_size=1 \
  --tasks tasks/bio_choicesonly.yaml \
  --batch_size auto --log_samples --use_cache cache \
  --output_path results/mistral-7b/choicesonly
```

Then see **scripts/** for consensus, abstention, heuristics, permutation, and run-all helpers.

## Tasks included

* `tasks/bio_mc.yaml` — conventional multiple choice
* `tasks/bio_choicesonly.yaml` — options only (no stem); use multi-model consensus
* `tasks/bio_cloze.yaml` — verified cloze; score each option as a continuation

## Data format

Each JSONL line:

```json
{"id":"abc123", "question":"Which letter comes first?", "choices":["A","C","B","D"], "answer": 0}
```

## Safety & privacy

* Local runs only; do not upload logs externally.
* Hash IDs before release (toy set here is harmless).
* See `docs/SAFETY.md`.

## Next steps

* Add your private datasets under `data/` (never commit sensitive text).
* Duplicate the YAMLs to make variants per domain.
* Use `scripts/run_all.sh` to run a model suite, then `vote_consensus.py`, `abstain_ci.py`, `heuristics.py`, and `build_permuted.py`.

