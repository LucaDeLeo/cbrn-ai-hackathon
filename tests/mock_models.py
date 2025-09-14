"""Mock fixtures for model operations to avoid loading real models in tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class MockTokenizer:
    """Lightweight mock tokenizer that doesn't load any models."""

    pad_token: str | None = None
    eos_token: str = "<eos>"

    def __call__(self, text: str, **kwargs) -> Any:
        """Mock tokenization - returns simple object."""
        return SimpleNamespace(
            input_ids=[[1, 2, 3]],  # Mock token IDs
            attention_mask=[[1, 1, 1]]
        )

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> MockTokenizer:
        """Mock loading - returns instance without downloading."""
        return cls()

    def decode(self, ids: list) -> str:
        """Mock decoding."""
        return "mock decoded text"


class MockModel:
    """Lightweight mock model that doesn't load any weights."""

    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.device = "cpu"

    def to(self, device: str) -> MockModel:
        """Mock device placement."""
        self.device = device
        return self

    def eval(self) -> None:
        """Mock eval mode."""
        pass

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> MockModel:
        """Mock loading - returns instance without downloading."""
        return cls(model_name)

    def __call__(self, **kwargs) -> SimpleNamespace:
        """Mock forward pass - returns simple loss."""
        return SimpleNamespace(
            loss=SimpleNamespace(item=lambda: 0.5),
            logits=[[0.1, 0.2, 0.3, 0.4]]
        )

    def generate(self, **kwargs) -> list:
        """Mock generation."""
        return [[1, 2, 3]]


def mock_score_cloze_options(
    model_name: str,
    stems: list[str],
    choices_list: list[list[str]],
    device: str = "cpu",
    dtype: str = "float32",
) -> tuple[list[int], list[float]]:
    """Mock cloze scoring without loading models."""
    preds = []
    confs = []

    for i, choices in enumerate(choices_list):
        # Simple deterministic mock: pick first choice
        pred_idx = i % len(choices)
        preds.append(pred_idx)
        # Mock confidence based on number of choices
        conf = 1.0 / len(choices) if choices else 0.5
        confs.append(conf)

    return preds, confs


def mock_run_cloze_hf(
    model: str,
    dataset_path: str,
    seed: int = 123,
    max_items: int | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    logs_dir: str = "logs",
) -> Path:
    """Mock HF cloze run without loading models."""
    # Create minimal mock results
    mock_results = {
        "task": "cloze_hf_logprob",
        "tags": ["cloze", "hf_logprob", "mock"],
        "model": model,
        "seed": seed,
        "results": [
            {
                "id": f"mock_{i}",
                "pred_index": i % 4,
                "confidence": 0.7 + (i % 3) * 0.1,
                "target_index": i % 4,
                "correct": True,
            }
            for i in range(min(max_items or 3, 3))
        ],
        "samples": []
    }

    # Write mock results
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(logs_dir) / f"mock_cloze__{model.replace('/', '_')}__seed{seed}.json"
    out_path.write_text(json.dumps(mock_results))
    return out_path


def mock_inspect_eval(**kwargs) -> dict:
    """Mock inspect_ai eval without running actual evaluation."""
    return {
        "model": kwargs.get("model", "mock-model"),
        "task": kwargs.get("task", "mock-task"),
        "results": {
            "accuracy": 0.85,
            "total": 10,
            "correct": 8
        },
        "samples": []
    }


# Mock torch if needed
class MockTorch:
    """Minimal torch mock to avoid importing real torch."""

    class Cuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def empty_cache() -> None:
            pass

    @staticmethod
    def tensor(data, **kwargs):
        return SimpleNamespace(item=lambda: data)

    @staticmethod
    def no_grad():
        class NoGradContext:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        return NoGradContext()

    float32 = "float32"
    bfloat16 = "bfloat16"
    long = "long"


def patch_all_models(monkeypatch):
    """Patch all model-related imports to use mocks."""
    import sys

    # Mock transformers
    mock_transformers = SimpleNamespace(
        AutoTokenizer=MockTokenizer,
        AutoModelForCausalLM=MockModel,
    )

    # Mock torch
    mock_torch_module = MockTorch()

    # Apply patches
    monkeypatch.setattr(sys.modules.get("transformers", sys), "AutoTokenizer", MockTokenizer, raising=False)
    monkeypatch.setattr(sys.modules.get("transformers", sys), "AutoModelForCausalLM", MockModel, raising=False)
    monkeypatch.setattr(sys.modules.get("torch", sys), "cuda", MockTorch.Cuda, raising=False)

    # Mock the actual functions
    if "robustcbrn.tasks.cloze_full" in sys.modules:
        monkeypatch.setattr("robustcbrn.tasks.cloze_full.score_cloze_options", mock_score_cloze_options)

    if "robustcbrn.tasks.cloze_logprob" in sys.modules:
        monkeypatch.setattr("robustcbrn.tasks.cloze_logprob.run_cloze_hf", mock_run_cloze_hf)

    return mock_transformers, mock_torch_module
