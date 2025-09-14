from __future__ import annotations

"""Inspect task for paraphrase consistency robustness.

Generates deterministic paraphrase variants for each item and evaluates with
the standard multiple choice solver. Logs include per-sample 'variant' and
'paraphrase_id' fields via sample metadata for downstream aggregation.
"""

from typing import Optional

from .common import load_mcq_dataset
from ..qa.paraphrase import generate_paraphrases

try:
    from inspect_ai import Task, task  # type: ignore
    from inspect_ai.dataset import MemoryDataset, Sample  # type: ignore
    from inspect_ai.scorer import choice  # type: ignore
    from inspect_ai.solver import multiple_choice  # type: ignore
except Exception:  # pragma: no cover
    Task = None  # type: ignore
    task = None  # type: ignore
    MemoryDataset = None  # type: ignore
    Sample = None  # type: ignore
    multiple_choice = None  # type: ignore
    choice = None  # type: ignore


def _build_paraphrase_dataset(ds, k: int) -> "MemoryDataset":  # type: ignore[name-defined]
    # Accept both MemoryDataset and list[dict] from loader
    try:
        samples = list(ds)  # type: ignore[arg-type]
    except Exception:
        samples = ds  # type: ignore[assignment]

    out = []
    for s in samples:
        if hasattr(s, "input"):
            stem = str(s.input)
            choices = list(s.choices)
            target = int(s.target)
            sid = getattr(s, "id", None)
            meta = getattr(s, "metadata", {}) or {}
        else:  # dict style
            stem = str(s["input"])  # type: ignore[index]
            choices = list(s["choices"])  # type: ignore[index]
            target = int(s["target"])  # type: ignore[index]
            sid = s.get("id")  # type: ignore[index]
            meta = s.get("metadata", {}) or {}  # type: ignore[index]

        variants = generate_paraphrases(stem, k=k)
        for p in variants:
            md = {
                **meta,
                "variant": p.variant,
                "paraphrase_id": p.variant if p.variant != "orig" else "orig",
            }
            out.append(
                Sample(
                    input=p.text,
                    choices=choices,
                    target=target,
                    id=str(sid),
                    metadata=md,
                )
            )
    return MemoryDataset(out)  # type: ignore[call-arg]


if task is not None:

    @task
    def paraphrase_consistency(
        dataset_path: str,
        k: int = 2,
        seed: int = 123,
        max_items: Optional[int] = None,
    ) -> Task:
        ds = load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=max_items)
        pds = _build_paraphrase_dataset(ds, k=k)
        return Task(
            dataset=pds,
            solver=multiple_choice(seed=seed),  # type: ignore[call-arg]
            scorer=choice(),  # type: ignore[misc]
            tags=["robustness", "paraphrase_consistency"],
        )

else:

    def paraphrase_consistency(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "inspect_ai is not available: cannot create paraphrase_consistency task."
        )

