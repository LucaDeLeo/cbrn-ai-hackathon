from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


def _env_list(name: str, default: str = "") -> List[str]:
    val = os.getenv(name, default)
    return [x.strip() for x in val.split(";") if x.strip()]


@dataclass
class ModelConfig:
    """Model configuration for local HF models and optional API backends.

    Defaults prefer local, instruction-tuned 7B–8B models that run on an A100.
    """

    local_models: List[str] = field(
        default_factory=lambda: _env_list(
            "MODELS",
            (
                "meta-llama/Llama-3.1-8B-Instruct;"
                "mistralai/Mistral-7B-Instruct-v0.3;"
                "Qwen/Qwen2.5-7B-Instruct"
            ),
        )
    )
    api_model: Optional[str] = os.getenv("INSPECT_EVAL_MODEL") or None
    device: str = os.getenv("DEVICE", "cuda")
    dtype: str = os.getenv("DTYPE", "bfloat16")
    max_seq_len: int = int(os.getenv("MAX_SEQ_LEN", "4096"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "4"))
    seeds: List[int] = field(
        default_factory=lambda: [
            int(x) for x in (_env_list("SEEDS", "123;456") or ["123", "456"])
        ]
    )


@dataclass
class BudgetConfig:
    cloud_budget_usd: float = float(os.getenv("CLOUD_BUDGET_USD", "400"))
    gpu_hourly_usd: Optional[float] = (
        float(os.getenv("GPU_HOURLY_USD", "0")) if os.getenv("GPU_HOURLY_USD") else None
    )
    api_budget_usd: float = float(os.getenv("API_BUDGET_USD", "0"))


@dataclass
class Paths:
    logs_dir: str = os.getenv("LOGS_DIR", "logs")
    results_dir: str = os.getenv("RESULTS_DIR", "artifacts/results")
    figs_dir: str = os.getenv("FIGS_DIR", "artifacts/figs")
    budget_dir: str = os.getenv("BUDGET_DIR", ".budget")


def get_model_config() -> ModelConfig:
    return ModelConfig()


def get_budget_config() -> BudgetConfig:
    return BudgetConfig()


def get_paths() -> Paths:
    return Paths()
