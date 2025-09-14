#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs" / "results" / "report.md"
SUMMARY_PATH = ROOT / "artifacts" / "results" / "summary.json"
ALL_RESULTS_CSV = ROOT / "artifacts" / "results" / "all_results.csv"
BUDGET_JSON = ROOT / ".budget" / "budget.json"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{100.0*float(x):.1f}%"
    except Exception:
        return "n/a"


def _load_summary() -> dict[str, Any]:
    if SUMMARY_PATH.exists():
        try:
            return json.loads(SUMMARY_PATH.read_text())
        except Exception:
            pass
    return {}


def _load_df() -> pd.DataFrame:
    if ALL_RESULTS_CSV.exists():
        try:
            return pd.read_csv(ALL_RESULTS_CSV)
        except Exception:
            pass
    return pd.DataFrame()


def _compute_overall_accuracy(df: pd.DataFrame) -> float | None:
    if df.empty or "correct" not in df.columns:
        return None
    try:
        return float(pd.to_numeric(df["correct"], errors="coerce").mean())
    except Exception:
        return None


def _compute_exploitable_pct(df: pd.DataFrame) -> float | None:
    if df.empty or "exploitable" not in df.columns or "id" not in df.columns:
        return None
    try:
        # Filter for mcq_choices_only tasks only
        if "task" in df.columns:
            sub = df[df["task"] == "mcq_choices_only"][["id", "exploitable"]].dropna()
        else:
            sub = df[["id", "exploitable"]].dropna()
        # One row per item id
        sub = sub.groupby("id").first().reset_index()
        if sub.empty:
            return None
        return float(pd.to_numeric(sub["exploitable"], errors="coerce").mean())
    except Exception:
        return None


def _collect_seeds(df: pd.DataFrame) -> list[int]:
    if df.empty or "seed" not in df.columns:
        return []
    try:
        seeds = sorted({int(s) for s in pd.to_numeric(df["seed"], errors="coerce").dropna().astype(int).tolist()})
        return seeds
    except Exception:
        return []


def _load_budget() -> tuple[float | None, float | None]:
    hours = None
    usd = None
    try:
        if BUDGET_JSON.exists():
            data = json.loads(BUDGET_JSON.read_text())
            hours = float(data.get("accumulated_hours", 0.0))
            hourly_env = os.getenv("GPU_HOURLY_USD")
            if hourly_env:
                usd = hours * float(hourly_env)
    except Exception:
        pass
    return hours, usd


def _fill_line(line: str, key: str, value_text: str) -> str:
    # Replace the part after the colon while preserving list dash and spacing
    pattern = rf"^(\s*[-] {re.escape(key)}\s*:)\s*.*$"
    repl = rf"\1 {value_text}"
    return re.sub(pattern, repl, line)


def _env_list(var: str) -> list[str]:
    v = os.getenv(var, "").strip()
    if not v:
        return []
    parts = [p.strip() for p in v.split(";") if p.strip()]
    return parts


def _build_key_config_from_env() -> str | None:
    device = os.getenv("DEVICE")
    dtype = os.getenv("DTYPE")
    bs = os.getenv("BATCH_SIZE")
    msl = os.getenv("MAX_SEQ_LEN")
    pieces: list[str] = []
    if device:
        pieces.append(f"device={device}")
    if dtype:
        pieces.append(f"dtype={dtype}")
    if bs:
        pieces.append(f"batch_size={bs}")
    if msl:
        pieces.append(f"max_seq_len={msl}")
    return "; ".join(pieces) if pieces else None


def main() -> int:
    summary = _load_summary()
    df = _load_df()

    overall_acc = _compute_overall_accuracy(df)
    exploitable_pct = _compute_exploitable_pct(df)

    mcq_gap = summary.get("mcq_vs_cloze", {}) if isinstance(summary.get("mcq_vs_cloze"), dict) else {}
    gap = mcq_gap.get("gap")
    ci_lo = mcq_gap.get("ci_lo")
    ci_hi = mcq_gap.get("ci_hi")

    abst = summary.get("abstention_overconfidence", {}) if isinstance(summary.get("abstention_overconfidence"), dict) else {}
    abst_rate = abst.get("abstention_rate")
    over_rate = abst.get("overconfidence_rate")

    heur = summary.get("heuristics_summary", {}) if isinstance(summary.get("heuristics_summary"), dict) else {}
    longest = heur.get("longest_answer_acc")
    pos_rate = heur.get("position_bias_rate")

    hours, gpu_usd = _load_budget()

    # Models from summary; seeds from df with env fallback
    models: list[str] = summary.get("models", []) if isinstance(summary.get("models"), list) else []
    if not models:
        models = _env_list("MODELS")
    seeds = _collect_seeds(df)
    if not seeds:
        try:
            seeds_env = [int(s) for s in _env_list("SEEDS")]
            seeds = sorted(set(seeds_env))
        except Exception:
            pass
    # Optional: model revisions
    revisions: list[str] = []
    if isinstance(summary.get("model_revisions"), list):
        revisions = [str(x) for x in summary.get("model_revisions", [])]
    if not revisions:
        revisions = _env_list("MODEL_REVISIONS")
    # Optional key config from env
    key_cfg = _build_key_config_from_env()

    if not REPORT_PATH.exists():
        print(f"[fill_report] Report file not found: {REPORT_PATH}")
        return 1

    lines = REPORT_PATH.read_text(encoding='utf-8').splitlines()
    joined = "\n".join(lines)
    out_lines: list[str] = []
    in_model_cards = False
    for i, ln in enumerate(lines):
        # Track whether we're inside the "Model Cards Used" bullet section
        if ln.strip().startswith("Model Cards Used"):
            in_model_cards = True
        elif in_model_cards and ln.strip() == "":
            in_model_cards = False
        if ln.strip().startswith("- Overall accuracy:"):
            ln = _fill_line(ln, "Overall accuracy", _fmt_pct(overall_acc))
        elif ln.strip().startswith("- Choices‑only consensus exploitable %:") or ln.strip().startswith("- Choices-only consensus exploitable %:"):
            ln = _fill_line(ln, "Choices‑only consensus exploitable %", _fmt_pct(exploitable_pct))
        elif ln.strip().startswith("- MCQ↔Cloze gap (95% CI):"):
            if gap is not None and ci_lo is not None and ci_hi is not None:
                ln = _fill_line(
                    ln,
                    "MCQ↔Cloze gap (95% CI)",
                    f"Δ={float(gap):.3f} (95% CI: [{float(ci_lo):.3f}, {float(ci_hi):.3f}])",
                )
        elif ln.strip().startswith("- Abstention / overconfidence:"):
            ln = _fill_line(
                ln,
                "Abstention / overconfidence",
                f"abst={_fmt_pct(abst_rate)}, overconf={_fmt_pct(over_rate)}",
            )
        elif ln.strip().startswith("- Runtime / cost:"):
            parts = []
            if hours is not None:
                parts.append(f"hours={hours:.2f}h")
            if gpu_usd is not None:
                parts.append(f"gpu≈${gpu_usd:.2f}")
            ln = _fill_line(ln, "Runtime / cost", ", ".join(parts) if parts else "n/a")
        elif in_model_cards and ln.strip().startswith("- Models:"):
            ln = _fill_line(ln, "Models", "; ".join(models) if models else "n/a")
        elif in_model_cards and ln.strip().startswith("- Seeds:"):
            ln = _fill_line(ln, "Seeds", "; ".join(str(s) for s in seeds) if seeds else "n/a")
        elif in_model_cards and ln.strip().startswith("- Revisions:"):
            ln = _fill_line(ln, "Revisions", "; ".join(revisions) if revisions else "TODO (e.g., HF snapshot hashes or provider revisions)")
        elif in_model_cards and ln.strip().startswith("- Key config:"):
            ln = _fill_line(ln, "Key config", key_cfg if key_cfg else "TODO (device=cuda; dtype=bfloat16; batch_size=4; max_seq_len=4096)")
        elif ln.strip().startswith("- longest‑answer accuracy:"):
            ln = _fill_line(ln, "longest‑answer accuracy", _fmt_pct(longest))
        elif ln.strip().startswith("- position‑bias rate (first/last):"):
            ln = _fill_line(ln, "position‑bias rate (first/last)", _fmt_pct(pos_rate))
        out_lines.append(ln)

    REPORT_PATH.write_text("\n".join(out_lines) + "\n", encoding='utf-8')
    print("[fill_report] Updated docs/results/report.md from artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
