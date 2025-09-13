# Determinism & Reproducibility

## Deterministic Mode Settings

```python
def enable_determinism(seed: int):
    import random, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Torch deterministic algorithms
    torch.use_deterministic_algorithms(True)

    # cuDNN / CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Disable TF32 for determinism
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Encourage deterministic cublas (set before torch init in CLI)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    # Disable non-deterministic optimizations if enabled elsewhere
    # (BetterTransformer/Flash-Attn are disabled when deterministic mode is on)
```

All stochastic components (bootstrap CI, permutations) draw RNG from a seeded generator derived from the top-level seed and component-specific offsets to ensure reproducible results across runs.
