# Setup on Lambda Cloud A100

1. Launch an instance with **Lambda Stack** (drivers/CUDA/torch preinstalled).
2. SSH in, install conda (if needed), then:

```bash
git clone <your-fork-url> robustcbrn-eval && cd robustcbrn-eval
conda env create -f environment.yml && conda activate robustcbrn
pip install -r requirements.txt
python - <<'PY'
import torch; print("CUDA OK:", torch.cuda.is_available())
PY
```

3. Run the toy tasks (see README). Prefer `--batch_size auto` and vLLM with
   `gpu_memory_utilization≈0.9` on 40GB A100.

