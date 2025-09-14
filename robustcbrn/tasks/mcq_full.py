from __future__ import annotations

from .common import load_mcq_dataset

try:
    from inspect_ai import Task, task  # type: ignore
    from inspect_ai.scorer import choice  # type: ignore
    from inspect_ai.solver import multiple_choice  # type: ignore
except Exception as e:  # pragma: no cover - import-time message for missing inspect
    Task = None  # type: ignore
    task = None  # type: ignore
    multiple_choice = None  # type: ignore
    choice = None  # type: ignore
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


if task is not None:

    @task
    def mcq_full(
        dataset_path: str,
        seed: int = 123,
        max_items: int | None = None,
    ) -> Task:
        """Baseline MCQ evaluation with Inspect's multiple_choice solver.

        No chain-of-thought; direct answer selection with choice() scorer.
        """

        ds = load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=max_items)
        return Task(
            dataset=ds,
            solver=multiple_choice(seed=seed),  # type: ignore[call-arg]
            scorer=choice(),  # type: ignore[misc]
            tags=["mcq", "full"],
        )

else:

    def mcq_full(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError(
            "inspect_ai is not available: cannot create mcq_full task."
        )
