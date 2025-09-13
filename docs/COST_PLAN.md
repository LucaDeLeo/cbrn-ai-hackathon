# Cost Plan ($400 Guard)

Budget variables (from `.env`):
- `CLOUD_BUDGET_USD` (default: 400) — cap for GPU time spend.
- `GPU_HOURLY_USD` — your provider’s hourly rate for the instance.
- `API_BUDGET_USD` — optional cap for API costs (off by default).

How estimates work:
- Projected hours ≈ models × seeds × items × per‑item time.
- Sample estimate: ~0.04s/item on an 8B model → ~0.000011h/item.
- BudgetGuard dry‑run computes `projected_hours * GPU_HOURLY_USD` and warns if exceeding.

Safe defaults:
- Subset sizes for early runs (e.g., 128–512 items).
- Batch size 4, bf16 on A100.
- 2 seeds.

Adjusting to fit budget:
- Reduce `SUBSET` in `make run`.
- Lower `BATCH_SIZE` or `DTYPE`.
- Fewer models/seeds.
- Prefer local models; leave `INSPECT_EVAL_MODEL` unset unless necessary.

Persistence:
- `.budget/budget.json` tracks accumulated hours and API spend locally. Delete to reset if needed (record totals elsewhere first).

