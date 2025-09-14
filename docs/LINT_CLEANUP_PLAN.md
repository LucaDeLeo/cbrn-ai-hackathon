# Ruff Lint Cleanup Plan

This repository currently has many Ruff findings across `robustcbrn` and `tests` (e.g., import order `I001`, module import position `E402`, unused imports `F401`, `B905` strict zip, etc.). To bring lint to green without destabilizing work, use an incremental plan:

- Scope and Baseline
  - Run `ruff --output-format=full` to snapshot the current error set.
  - Keep existing `make lint` target (`ruff check robustcbrn tests`) to preserve CI expectations.

- Phase 1: Safe Auto-fixes
  - Apply safe autofixes with `ruff --fix` on narrow rule sets project-wide:
    - `I001` (import sort/format)
    - `F401` (unused imports)
    - `B905` (zip strict)
  - Verify tests after each batch. Avoid touching behavioral code beyond import ordering and unused removals.

- Phase 2: E402 (module-level import position)
  - Move imports to file tops where feasible.
  - If top-level imports are intentionally delayed (e.g., avoid heavy deps at import time), locally `# noqa: E402` with a brief comment.

- Phase 3: Targeted Remaining Rules
  - Address remaining rule violations per module, prioritizing high-churn files last to reduce merge risk.
  - For test files, consider softening a subset of stylistic rules in `pyproject.toml` if they conflict with established testing patterns (e.g., allowing inline imports in tests).

- Pre-commit Integration
  - Add a `pre-commit` config to run `ruff` (check + fix) on staged files.
  - Optional: Enable `ruff format` to standardize formatting alongside checks.

- CI & Quality Gates
  - Add a GitHub action (or extend existing) to run `make lint` and `make test` on PRs.
  - Consider a temporary allowlist of rule codes in CI to gradually tighten over time.

- Tracking & Stories
  - Create a small epic with subtasks per phase/module.
  - Gate each subtask on “no behavior changes” and “tests green”.

Notes
- This plan avoids changing project-level behavior. Any rule relaxations should be limited, justified, and documented.
- If we decide to adjust rules, prefer scoping changes to tests or specific modules rather than broad global ignores.
