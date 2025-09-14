from __future__ import annotations

"""Inspect task for perturbation stability robustness.

Generates deterministic perturbation variants for each item and evaluates with
the standard multiple choice solver. Per-sample metadata includes 'variant' and
'perturbation_kind' for aggregation.
"""


from ..qa.perturb import generate_perturbations
from .common import load_mcq_dataset

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


def _build_perturb_dataset(ds, k: int) -> MemoryDataset:  # type: ignore[name-defined]
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

        variants = generate_perturbations(stem, choices, target, k=k)
        for p in variants:
            md = {
                **meta,
                "variant": p.variant,
                "perturbation_kind": p.kind,
            }
            out.append(
                Sample(
                    input=p.stem,
                    choices=p.choices,
                    target=p.target_index,
                    id=str(sid),
                    metadata=md,
                )
            )
    return MemoryDataset(out)  # type: ignore[call-arg]


if task is not None:

    @task
    def perturbation_stability(
        dataset_path: str,
        k: int = 3,
        seed: int = 123,
        max_items: int | None = None,
    ) -> Task:
        ds = load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=max_items)
        pds = _build_perturb_dataset(ds, k=k)
        return Task(
            dataset=pds,
            solver=multiple_choice(seed=seed),  # type: ignore[call-arg]
            scorer=choice(),  # type: ignore[misc]
            tags=["robustness", "perturbation_stability"],
        )

else:

    def perturbation_stability(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "inspect_ai is not available: cannot create perturbation_stability task."
        )

