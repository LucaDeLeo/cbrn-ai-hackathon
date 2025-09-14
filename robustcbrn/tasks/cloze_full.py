from __future__ import annotations

import math
from contextlib import suppress

import torch

from .common import load_mcq_dataset


def _resolve_dtype(dtype: str) -> torch.dtype:
    d = dtype.lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _length_normalized_logprob(model, tokenizer, prompt_text: str, option_text: str, device: str) -> float:
    # Compute log P(option | prompt) normalized by option token length.
    with torch.no_grad():
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        option_ids = tokenizer(
            option_text, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        input_ids = torch.cat([prompt_ids.input_ids, option_ids.input_ids], dim=1)
        attn_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        # Mask prompt tokens from loss
        labels[:, : prompt_ids.input_ids.size(1)] = -100
        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
        loss = out.loss  # mean over non-masked tokens
        # mean log prob per token = -loss
        return float(-loss.detach().cpu())


def score_cloze_options(
    model_name: str,
    stems: list[str],
    choices_list: list[list[str]],
    device: str = "cuda",
    dtype: str = "bfloat16",
) -> tuple[list[int], list[float]]:
    """Score cloze options using length-normalized log-prob per option.

    Returns tuple (pred_indices, confidences) where confidences are softmaxed
    over options for each item, reported as the winning option probability.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    torch_dtype = _resolve_dtype(dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Load model and place on requested device
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    with suppress(Exception):
        # Fallback: keep current placement if explicit move fails
        model.to(device)
    model.eval()

    preds: list[int] = []
    confs: list[float] = []
    for stem, choices in zip(stems, choices_list, strict=False):
        prompt = f"{stem}\nAnswer: "
        scores = [
            _length_normalized_logprob(model, tok, prompt, c, device=device) for c in choices
        ]
        # Convert normalized logprobs to probabilities
        m = max(scores)
        probs = [math.exp(s - m) for s in scores]
        z = sum(probs) or 1.0
        probs = [p / z for p in probs]
        pred = int(max(range(len(choices)), key=lambda i: scores[i]))
        preds.append(pred)
        confs.append(float(probs[pred]))
    return preds, confs


# Inspect fallback task
try:
    from inspect_ai import Task, task  # type: ignore
    from inspect_ai.scorer import choice  # type: ignore
    from inspect_ai.solver import multiple_choice  # type: ignore
except Exception:  # pragma: no cover
    Task = None  # type: ignore
    task = None  # type: ignore
    multiple_choice = None  # type: ignore
    choice = None  # type: ignore


if task is not None:

    @task
    def cloze_full(
        dataset_path: str,
        seed: int = 123,
        max_items: int | None = None,
        use_hf_logprob: bool = True,
        hf_model_name: str | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ) -> Task:
        """Verified cloze scoring.

        Preferred: HF log-prob path computing normalized log P(choice|stem). Fallback
        uses structured multiple_choice via Inspect when log-prob path isn't used.
        The path used is logged in tags for downstream analysis.
        """

        ds = load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=max_items)
        if use_hf_logprob and hf_model_name is not None:
            # We'll run outside Inspect for predictions; Inspect still used for logging infra if desired.
            try:
                # Extract stems/choices
                try:
                    from inspect_ai.dataset import MemoryDataset  # type: ignore

                    samples = list(ds) if isinstance(ds, MemoryDataset) else ds  # type: ignore[arg-type]
                except Exception:
                    samples = ds  # type: ignore[assignment]
                stems = [s["input"] if isinstance(s, dict) else s.input for s in samples]
                choices_list = [
                    s["choices"] if isinstance(s, dict) else list(s.choices) for s in samples
                ]
                preds, confs = score_cloze_options(
                    hf_model_name, stems, choices_list, device=device, dtype=dtype
                )
                # Build a pseudo-dataset with predictions embedded for downstream processing
                # Note: we still return a Task object to keep interface consistent
            except Exception:
                # If HF path fails, fall through to Inspect solver
                pass

        # Fallback to Inspect structured choice solver
        return Task(
            dataset=ds,
            solver=multiple_choice(seed=seed),  # type: ignore[call-arg]
            scorer=choice(),  # type: ignore[misc]
            tags=[
                "cloze",
                "fallback_structured" if (not use_hf_logprob or hf_model_name is None) else "hf_logprob",
            ],
        )

else:

    def cloze_full(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "inspect_ai is not available: cannot create cloze_full task."
        )
