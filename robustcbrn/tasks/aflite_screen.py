from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .common import load_mcq_dataset
from ..config import get_paths
from ..qa.filters import aflite_lite, aflite_join_with_items


def run_aflite_screen(
    dataset_path: str,
    tau: float = 0.7,
    k_folds: int = 5,
    seed: int = 123,
    max_items: Optional[int] = None,
    out_path: Optional[str] = None,
) -> str:
    """Run AFLite-lite screen and write an Inspect-like JSON log.

    The output JSON is compatible with the aggregator (task='aflite_screen').
    """
    ds = load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=max_items)
    # Convert Inspect MemoryDataset to simple iterable of dicts if necessary
    try:
        samples = list(ds)  # type: ignore[arg-type]
    except Exception:
        samples = ds  # type: ignore[assignment]

    results = aflite_lite(samples, tau=tau, k_folds=k_folds, seed=seed)
    joined = aflite_join_with_items(samples, results)

    payload = {
        "task": "aflite_screen",
        "model": "aflite-lite",
        "seed": seed,
        "samples": [
            {
                "id": rec["id"],
                "predictability_score": rec.get("predictability_score"),
                "flag_predictable": rec.get("flag_predictable"),
                "probe_hit": rec.get("probe_hit", []),
            }
            for rec in joined
        ],
    }

    out_dir = Path(get_paths().logs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        base = Path(dataset_path).stem
        out_path = str(out_dir / f"aflite_screen_{base}.json")
    Path(out_path).write_text(json.dumps(payload, indent=2))
    return out_path


def cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AFLite-lite adversarial filtering screen")
    ap.add_argument("dataset_path", help="Path to MCQ JSONL dataset")
    ap.add_argument("--tau", type=float, default=0.7, help="Predictability threshold")
    ap.add_argument("--k-folds", type=int, default=5, help="Cross-validation folds")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--out", type=str, default=None, help="Explicit output JSON path")
    args = ap.parse_args(argv)
    out = run_aflite_screen(
        dataset_path=args.dataset_path,
        tau=args.tau,
        k_folds=args.k_folds,
        seed=args.seed,
        max_items=args.max_items,
        out_path=args.out,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

