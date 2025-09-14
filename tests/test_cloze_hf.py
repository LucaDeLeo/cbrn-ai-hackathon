from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


def _truthy_env(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def test_hf_cloze_importable() -> None:
    # Always ensure modules/functions import cleanly
    from robustcbrn.tasks.cloze_full import score_cloze_options  # noqa: F401
    from robustcbrn.tasks.cloze_logprob import run_cloze_hf  # noqa: F401


@pytest.mark.skipif(
    not _truthy_env(os.getenv("RUN_HF_CLOZE_SMOKE")),
    reason="HF cloze smoke disabled (set RUN_HF_CLOZE_SMOKE=1 to enable)",
)
def test_hf_cloze_smoke_mocked(monkeypatch) -> None:
    # Exercise score_cloze_options with mocked HF components to avoid network/weights
    import torch
    import transformers  # type: ignore

    from robustcbrn.tasks.cloze_full import score_cloze_options

    class DummyBatch:
        def __init__(self, text: str):
            # Length-scaled tokenization to make options differ
            n = max(1, min(8, len(text) // 5))
            self.input_ids = torch.arange(n, dtype=torch.long).view(1, n)

        def to(self, device: str) -> DummyBatch:  # noqa: ARG002
            return self

    class DummyTokenizer:
        pad_token: str | None = None
        eos_token: str = "<eos>"

        def __call__(self, text: str, add_special_tokens: bool, return_tensors: str) -> DummyBatch:  # noqa: ARG002
            assert return_tensors == "pt"
            return DummyBatch(text)

        @classmethod
        def from_pretrained(cls, model_name: str) -> DummyTokenizer:  # noqa: ARG002
            return cls()

    class DummyModel:
        def to(self, device: str) -> DummyModel:  # noqa: ARG002
            return self

        def eval(self) -> None:
            return None

        @classmethod
        def from_pretrained(cls, model_name: str, torch_dtype: torch.dtype) -> DummyModel:  # noqa: ARG002
            return cls()

        def __call__(self, *, input_ids, attention_mask, labels):  # noqa: ANN001, D401
            # Compute a deterministic loss based on the number of unmasked tokens
            # (i.e., option length), so different options produce different scores.
            opt_len = int((labels != -100).sum().item())
            loss = torch.tensor(1.0 / (opt_len + 1.0), dtype=torch.float32)
            return SimpleNamespace(loss=loss)

    # Patch transformers API used by score_cloze_options
    monkeypatch.setattr(transformers, "AutoTokenizer", DummyTokenizer, raising=True)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", DummyModel, raising=True)

    stems = [
        "The capital of France is _____ .",
        "2 + 2 equals _____",
    ]
    choices_list = [["Berlin", "Paris", "Rome"], ["4", "5"]]

    preds, confs = score_cloze_options(
        model_name="dummy-model",
        stems=stems,
        choices_list=choices_list,
        device="cpu",
        dtype="float32",
    )

    # Shape and type checks
    assert isinstance(preds, list) and isinstance(confs, list)
    assert len(preds) == len(stems) == len(confs)
    for _i, (p, c, opts) in enumerate(zip(preds, confs, choices_list, strict=False)):
        assert isinstance(p, int) and 0 <= p < len(opts)
        assert isinstance(c, float) and 0.0 <= c <= 1.0


def test_run_cloze_hf_emits_log(tmp_path, monkeypatch) -> None:
    # Validate that run_cloze_hf writes a well-formed Inspect-like log envelope
    import json

    from robustcbrn.tasks import cloze_logprob as clp

    samples = [
        {"id": "a", "input": "2+2 is ___", "choices": ["3", "4"], "target": 1},
        {"id": "b", "input": "The sky is ___", "choices": ["blue"], "target": 0},
    ]

    def fake_load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=None):  # noqa: ARG001
        return samples

    def fake_score(model, stems, choices_list, device="cpu", dtype="float32"):  # noqa: ARG001
        assert len(stems) == len(choices_list) == len(samples)
        return [1, 0], [0.85, 1.0]

    monkeypatch.setattr(clp, "load_mcq_dataset", fake_load_mcq_dataset, raising=True)
    monkeypatch.setattr(clp, "score_cloze_options", fake_score, raising=True)

    out_path = clp.run_cloze_hf(
        model="dummy-model",
        dataset_path="unused.jsonl",
        device="cpu",
        dtype="float32",
        logs_dir=str(tmp_path),
    )

    data = json.loads(out_path.read_text())
    # Envelope keys
    assert {"task", "tags", "model", "seed"}.issubset(data.keys())
    # Results present with expected length and fields
    results = data.get("results") or data.get("samples")
    assert isinstance(results, list)
    assert len(results) == len(samples)
    for r in results:
        assert {"id", "pred_index", "confidence"}.issubset(r.keys())
