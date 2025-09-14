from __future__ import annotations

from .common import load_mcq_dataset, to_choices_only

try:
    from inspect_ai import Task, task  # type: ignore
    from inspect_ai.dataset import MemoryDataset, Sample  # type: ignore
    from inspect_ai.scorer import choice  # type: ignore
    from inspect_ai.solver import multiple_choice  # type: ignore
except Exception as e:  # pragma: no cover
    Task = None  # type: ignore
    task = None  # type: ignore
    multiple_choice = None  # type: ignore
    choice = None  # type: ignore
    MemoryDataset = None  # type: ignore
    Sample = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


def _build_choices_only_dataset(ds) -> MemoryDataset:  # type: ignore[name-defined]
    if MemoryDataset is None or Sample is None:
        raise RuntimeError("inspect_ai not available")
    if isinstance(ds, MemoryDataset):  # type: ignore[arg-type]
        samples = [Sample(**to_choices_only(s)) for s in ds]
        return MemoryDataset(samples)
    # assume list of dicts
    samples = [Sample(**to_choices_only(s)) for s in ds]
    return MemoryDataset(samples)


if task is not None:

    @task
    def mcq_choices_only(
        dataset_path: str,
        seed: int = 123,
        max_items: int | None = None,
    ) -> Task:
        """Choices-only MCQ screen: strip question stem; same solver + scorer.

        Emits per-item results suitable for consensus detection later.
        """

        ds = load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=max_items)
        ds_co = _build_choices_only_dataset(ds)
        return Task(
            dataset=ds_co,
            solver=multiple_choice(seed=seed),  # type: ignore[call-arg]
            scorer=choice(),  # type: ignore[misc]
            tags=["mcq", "choices_only"],
        )

else:

    def mcq_choices_only(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "inspect_ai is not available: cannot create mcq_choices_only task."
        )
