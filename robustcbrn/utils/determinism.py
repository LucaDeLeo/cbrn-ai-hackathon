from __future__ import annotations

import os
import random
from typing import Optional

np = None  # lazy import


def set_determinism(
    seed: int = 42,
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
    cublas_workspace: str = ":4096:8",
    python_hash_seed: int = 0,
    tokenizers_parallelism: bool = False,
) -> None:
    """Apply determinism controls across Python/NumPy/(optional)Torch.

    - Sets env vars for CUDNN/CUBLAS/PYTHONHASHSEED/TOKENIZERS_PARALLELISM
    - Seeds Python `random` and NumPy RNGs
    - If torch is available, seeds torch and enables deterministic algorithms
    """
    # Env vars
    os.environ["CUDNN_DETERMINISTIC"] = "1" if cudnn_deterministic else "0"
    os.environ["CUDNN_BENCHMARK"] = "1" if cudnn_benchmark else "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace
    os.environ["PYTHONHASHSEED"] = str(python_hash_seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false" if not tokenizers_parallelism else "true"

    # Python/NumPy
    random.seed(seed)
    global np
    try:
        if np is None:
            import numpy as _np  # type: ignore
            np = _np
        np.random.seed(seed)
    except Exception:
        pass

    # Optional: Torch
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Older versions may not support; ignore gracefully
            pass
        try:
            # For cuDNN controls via torch.backends
            import torch.backends.cudnn as cudnn

            cudnn.deterministic = cudnn_deterministic
            cudnn.benchmark = cudnn_benchmark
        except Exception:
            pass
    except Exception:
        # Torch not installed or failed to configure: acceptable for CPU-only environments
        pass
