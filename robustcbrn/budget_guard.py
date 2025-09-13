from __future__ import annotations

import argparse
import atexit
import json
import time
from contextlib import ContextDecorator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import get_budget_config, get_paths


@dataclass
class BudgetState:
    accumulated_hours: float = 0.0
    accumulated_api_usd: float = 0.0


def _budget_file() -> Path:
    paths = get_paths()
    p = Path(paths.budget_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p / "budget.json"


def load_budget_state() -> BudgetState:
    f = _budget_file()
    if f.exists():
        try:
            data = json.loads(f.read_text())
            return BudgetState(
                accumulated_hours=float(data.get("accumulated_hours", 0.0)),
                accumulated_api_usd=float(data.get("accumulated_api_usd", 0.0)),
            )
        except Exception:
            pass
    return BudgetState()


def save_budget_state(state: BudgetState) -> None:
    f = _budget_file()
    f.write_text(json.dumps(state.__dict__, indent=2))


class BudgetGuard(ContextDecorator):
    """Context manager/decorator to track and cap cloud spend.

    - Estimates GPU cost from wall-time * GPU hourly rate.
    - Optionally accumulates API spend (if provided by caller).
    - Aborts when projected cost would exceed remaining budget.
    """

    def __init__(
        self,
        job_name: str,
        hourly_usd: Optional[float] = None,
        cloud_budget_usd: Optional[float] = None,
        api_budget_usd: Optional[float] = None,
        dry_run: bool = False,
        projected_hours: Optional[float] = None,
        projected_api_usd: float = 0.0,
        verbose: bool = True,
    ) -> None:
        cfg = get_budget_config()
        self.job_name = job_name
        self.hourly_usd = hourly_usd if hourly_usd is not None else cfg.gpu_hourly_usd
        self.cloud_budget_usd = (
            cloud_budget_usd if cloud_budget_usd is not None else cfg.cloud_budget_usd
        )
        self.api_budget_usd = api_budget_usd if api_budget_usd is not None else cfg.api_budget_usd
        self.dry_run = dry_run
        self.projected_hours = projected_hours
        self.projected_api_usd = projected_api_usd
        self.verbose = verbose
        self._t0: Optional[float] = None
        self._state = load_budget_state()
        self._aborted = False

        if self.verbose:
            hourly = self.hourly_usd if self.hourly_usd else "unset"
            print(
                f"[budget] job='{self.job_name}' budget=${self.cloud_budget_usd:.2f} "
                f"hourly={hourly} api_budget=${self.api_budget_usd:.2f}"
            )

        if self.dry_run:
            self._dry_run_projection()

    def _hours_remaining(self) -> Optional[float]:
        if not self.hourly_usd or self.hourly_usd <= 0:
            return None
        remaining_usd = max(
            0.0, self.cloud_budget_usd - self._state.accumulated_hours * self.hourly_usd
        )
        return remaining_usd / self.hourly_usd if self.hourly_usd else None

    def _dry_run_projection(self) -> None:
        hours = self.projected_hours or 0.0
        api_usd = self.projected_api_usd
        cloud_usd = hours * (self.hourly_usd or 0)
        total_cloud_usd = self._state.accumulated_hours * (self.hourly_usd or 0) + cloud_usd
        total_api_usd = self._state.accumulated_api_usd + api_usd
        if self.verbose:
            print(
                f"[budget] DRY-RUN: project hours={hours:.3f} (${cloud_usd:.2f}), api=${api_usd:.2f}; "
                f"post-run totals: cloud=${total_cloud_usd:.2f} api=${total_api_usd:.2f}"
            )
        if (
            (self.cloud_budget_usd and self.hourly_usd and total_cloud_usd > self.cloud_budget_usd)
            or (self.api_budget_usd and total_api_usd > self.api_budget_usd)
        ):
            print("[budget] DRY-RUN would exceed budget. Consider:")
            print("  - Reduce dataset size (subset)")
            print("  - Lower batch size or precision")
            print("  - Use fewer seeds or models")

    def __enter__(self):
        if self.dry_run:
            # In dry-run we don't measure wall-time; return a no-op guard.
            return self
        # Pre-run projection if provided
        if self.projected_hours is not None:
            hours_rem = self._hours_remaining()
            if hours_rem is not None and self.projected_hours > hours_rem:
                print(
                    f"[budget] Aborting '{self.job_name}': projected {self.projected_hours:.2f}h "
                    f"> remaining {hours_rem:.2f}h"
                )
                self._aborted = True
                raise SystemExit(2)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.dry_run or self._aborted:
            return False
        t1 = time.perf_counter()
        elapsed_h = (t1 - (self._t0 or t1)) / 3600.0
        cloud_usd = elapsed_h * (self.hourly_usd or 0)
        if self.verbose:
            print(
                f"[budget] Completed '{self.job_name}' in {elapsed_h:.3f}h, cost ~${cloud_usd:.2f}"
            )

        # Project end-state during the run (if failure, still charge elapsed time)
        new_hours = self._state.accumulated_hours + elapsed_h
        if self.cloud_budget_usd and self.hourly_usd:
            total_usd = new_hours * self.hourly_usd
            if total_usd > self.cloud_budget_usd:
                print(
                    f"[budget] WARNING: Budget exceeded after this run: ${total_usd:.2f} > "
                    f"${self.cloud_budget_usd:.2f}"
                )
        self._state.accumulated_hours = new_hours
        save_budget_state(self._state)
        return False


def budget_guard_cli(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Budget guard for RobustCBRN evals")
    parser.add_argument("job", help="Job name (for logging)")
    parser.add_argument("--projected-hours", type=float, default=None, help="Projected job hours")
    parser.add_argument("--hourly-usd", type=float, default=None, help="GPU hourly USD rate")
    parser.add_argument("--api-usd", type=float, default=0.0, help="Projected API spend USD")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run projection only")
    args = parser.parse_args(argv)

    guard = BudgetGuard(
        job_name=args.job,
        hourly_usd=args.hourly_usd,
        dry_run=args.dry_run,
        projected_hours=args.projected_hours,
        projected_api_usd=args.api_usd,
    )
    if args.dry_run:
        return 0
    with guard:
        # No-op body; this CLI is for projection + begin/end accounting
        pass
    return 0


def _register_at_exit_note() -> None:
    def _note() -> None:
        ts = datetime.utcnow().isoformat()
        st = load_budget_state()
        print(
            f"[budget] {ts} totals -> hours={st.accumulated_hours:.3f} "
            f"api=${st.accumulated_api_usd:.2f}"
        )

    atexit.register(_note)


if __name__ == "__main__":
    _register_at_exit_note()
    raise SystemExit(budget_guard_cli())
