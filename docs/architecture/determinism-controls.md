# Determinism Controls

To achieve low run-to-run variance and reproducibility:
- Set seeds: `random.seed(SEED)`, `np.random.seed(SEED)`, `torch.manual_seed(SEED)`, `torch.cuda.manual_seed_all(SEED)`
- PyTorch: `torch.use_deterministic_algorithms(True)` (PyTorch 2.x)
- cuDNN: `CUDNN_DETERMINISTIC=1`, `CUDNN_BENCHMARK=0`
- cuBLAS: `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`)
- Python hashing: `PYTHONHASHSEED=0`
- Tokenizers: `TOKENIZERS_PARALLELISM=false`
- Disable CPU/GPU nondeterministic ops; sort data structures; enforce stable iteration order

Document all settings in the Reproducibility Bundle.
