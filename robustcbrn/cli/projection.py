from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import get_budget_config
from ..utils.cost import per_item_hours, suggest_subset


def print_projection(
    models: int,
    seeds: int,
    subset: int,
    hourly: float,
    budget_usd: float,
    budget_dir: str,
    safety: float = 0.9,
) -> None:
    total_items = models * seeds * subset
    print(f"[run_evalset] Workload: models×seeds×items = {models}×{seeds}×{subset} = {total_items}")

    if hourly and hourly > 0:
        # Load accumulated hours from budget state
        # Respect the provided budget_dir by temporarily overriding config lookup
        state_path = Path(budget_dir) / "budget.json"
        accum_hours = 0.0
        try:
            if state_path.exists():
                data = json.loads(state_path.read_text())
                accum_hours = float(data.get("accumulated_hours", 0.0))
        except Exception:
            pass

        total_budget_h = budget_usd / hourly if hourly else 0.0
        remaining_h = max(0.0, total_budget_h - accum_hours)
        print(
            f"[run_evalset] Budget: ${budget_usd:.2f} @ ${hourly:.2f}/h → remaining ≈ {remaining_h:.2f}h (used {accum_hours:.2f}h)"
        )
        if models > 0 and seeds > 0:
            per_h = per_item_hours()
            suggested = suggest_subset(remaining_h, models, seeds, safety=safety, per_item_h=per_h)
            if suggested > 0:
                print(f"[run_evalset] Suggestion: SUBSET≈{suggested} to fit budget ({int(safety*100)}% of remaining)")
            else:
                print(
                    "[run_evalset] Suggestion: remaining budget is ~0h → reduce models/seeds or increase budget"
                )
    else:
        print(
            "[run_evalset] Tip: set GPU_HOURLY_USD and CLOUD_BUDGET_USD in .env for budget-based SUBSET suggestions"
        )

    print(
        "[run_evalset] Tuning levers: SUBSET, MODELS, SEEDS, batch/precision (BATCH_SIZE/DTYPE)"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m robustcbrn.cli.projection",
        description="Print workload and budget-based projection guidance for run_evalset.",
    )
    ap.add_argument("--models", type=int, required=True)
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--subset", type=int, required=True)
    ap.add_argument("--hourly", type=float, default=0.0)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--budget-dir", type=str, default=None)
    ap.add_argument("--safety", type=float, default=0.9)
    args = ap.parse_args(argv)

    cfg = get_budget_config()
    budget_usd = args.budget_usd if args.budget_usd is not None else cfg.cloud_budget_usd
    budget_dir = args.budget_dir if args.budget_dir is not None else ".budget"

    print_projection(
        models=args.models,
        seeds=args.seeds,
        subset=args.subset,
        hourly=args.hourly,
        budget_usd=budget_usd,
        budget_dir=budget_dir,
        safety=args.safety,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
