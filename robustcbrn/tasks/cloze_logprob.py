from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Optional

from .common import load_mcq_dataset
from .cloze_full import score_cloze_options


def _sanitize_filename(s: str) -> str:
    return (
        s.replace("/", "-")
        .replace(":", "-")
        .replace(" ", "_")
        .replace("\\", "-")
    )


def _iter_samples(ds: Any) -> Iterable[dict]:
    """Yield simple dict samples with keys: id, input, choices, target.

    Supports either Inspect MemoryDataset/Sample or plain dict/list.
    """
    try:
        # Inspect dataset types
        from inspect_ai.dataset import MemoryDataset  # type: ignore

        if isinstance(ds, MemoryDataset):
            for s in ds:
                yield {
                    "id": getattr(s, "id", None),
                    "input": getattr(s, "input", ""),
                    "choices": list(getattr(s, "choices", []) or []),
                    "target": int(getattr(s, "target", -1)),
                }
            return
    except Exception:
        pass

    # Fallback: list[dict] path
    for s in ds:  # type: ignore[assignment]
        yield {
            "id": s.get("id"),
            "input": s.get("input", ""),
            "choices": list(s.get("choices", []) or []),
            "target": int(s.get("target", -1)),
        }


def run_cloze_hf(
    model: str,
    dataset_path: str,
    seed: int = 123,
    max_items: Optional[int] = None,
    device: str = "cuda",
    dtype: str = "bfloat16",
    logs_dir: str = "logs",
) -> Path:
    """Run HF length-normalized log-prob cloze scoring and emit Inspect-like log."""
    ds = load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=max_items)
    samples = list(_iter_samples(ds))
    stems = [s["input"] for s in samples]
    choices_list = [s["choices"] for s in samples]

    pred_indices, confidences = score_cloze_options(
        model, stems, choices_list, device=device, dtype=dtype
    )

    out_records = []
    for s, pred, conf in zip(samples, pred_indices, confidences, strict=False):
        target = int(s.get("target", -1))
        rec = {
            "id": str(s.get("id")),
            "pred_index": int(pred),
            "confidence": float(conf),
            "target_index": target,
            "correct": bool(int(pred) == target) if target >= 0 else None,
        }
        out_records.append(rec)

    # Root-level Inspect-compatible envelope
    model_s = _sanitize_filename(model)
    task_name = "cloze_hf_logprob"
    payload = {
        "task": task_name,
        "tags": ["cloze", "hf_logprob"],
        "model": model,
        "seed": seed,
        "results": out_records,  # aggregator accepts several keys
        "samples": out_records,
    }

    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(logs_dir) / f"{task_name}__{model_s}__seed{seed}.json"
    out_path.write_text(json.dumps(payload))
    return out_path


def cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="HF log-prob cloze scoring (Inspect-compatible logging)"
    )
    ap.add_argument("--model", required=True, help="HF model name or path")
    ap.add_argument("--dataset_path", required=True, help="Path to dataset JSONL")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--logs_dir", default="logs")
    args = ap.parse_args(argv)

    out = run_cloze_hf(
        model=args.model,
        dataset_path=args.dataset_path,
        seed=args.seed,
        max_items=args.max_items,
        device=args.device,
        dtype=args.dtype,
        logs_dir=args.logs_dir,
    )
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

