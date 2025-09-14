#!/usr/bin/env bash
set -euo pipefail
if [ -f .env ]; then
  set -a; source .env; set +a
fi

IFS=';' read -ra MODELS_ARR <<< "${MODELS:-meta-llama/Llama-3.1-8B-Instruct}"

echo "[download_models] Downloading/tokenizing models into HF cache"
echo "[download_models] Cache location: HF_HOME=${HF_HOME:-unset} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-unset}"
echo "[download_models] Tip: ensure ~15–20 GB per 7B–8B model available on the cache disk"
.venv/bin/python - <<'PY'
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

models = os.environ.get("MODELS", "").split(";")
for m in [x.strip() for x in models if x.strip()]:
    print(f"[download_models] -> {m}")
    try:
        tok = AutoTokenizer.from_pretrained(m)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _ = AutoModelForCausalLM.from_pretrained(m)
        print(f"[download_models] ok: {m}")
    except Exception as e:
        print(f"[download_models] warn: could not download {m}: {e}")
PY

echo "[download_models] Done."
