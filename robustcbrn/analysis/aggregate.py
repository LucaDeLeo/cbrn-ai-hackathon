from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import get_paths
from ..utils.stats import bootstrap_ci


@dataclass
class SampleResult:
    id: str
    task: str
    model: str
    correct: Optional[bool]
    pred_index: Optional[int]
    target_index: Optional[int]
    confidence: Optional[float]
    seed: Optional[int]
    # Optional QA/robustness fields
    flag_predictable: Optional[bool] = None
    predictability_score: Optional[float] = None
    probe_hit: Optional[str] = None
    # Ambiguity audit fields
    ambiguity_label: Optional[str] = None
    reason_codes: Optional[str] = None
    # Variant bookkeeping for future robustness modules
    variant: Optional[str] = None
    paraphrase_id: Optional[str] = None
    perturbation_kind: Optional[str] = None


def _collect_log_files(logs_dir: Path) -> list[Path]:
    return [p for p in logs_dir.rglob("*.json") if p.is_file()]


def _parse_inspect_log(path: Path) -> list[SampleResult]:
    data = json.loads(path.read_text())
    results: list[SampleResult] = []
    model = data.get("model") or data.get("provider_model") or "unknown"
    task = data.get("task") or data.get("task_name") or path.stem
    seed = data.get("seed")
    # Inspect formats can vary; attempt to find per-sample records
    samples = (
        data.get("samples")
        or data.get("records")
        or data.get("results")
        or data.get("items")
        or []
    )
    for s in samples:
        sid = str(s.get("id") or s.get("sample_id") or s.get("uid") or "")
        pred_idx = s.get("pred_index") or s.get("prediction_index")
        target_idx = s.get("target") if isinstance(s.get("target"), int) else s.get("target_index")
        correct = s.get("correct")
        conf = s.get("confidence")
        if correct is None and (pred_idx is not None and target_idx is not None):
            correct = bool(int(pred_idx) == int(target_idx))
        # Optional fields
        flag_predictable = s.get("flag_predictable")
        predictability_score = s.get("predictability_score")
        probe_hit = s.get("probe_hit")
        if isinstance(probe_hit, list):
            probe_hit = ",".join([str(x) for x in probe_hit])
        # Additional optional metadata fields
        ambiguity_label = s.get("label") or s.get("ambiguity_label")
        reason_codes = s.get("reason_codes")
        if isinstance(reason_codes, list):
            reason_codes = ",".join([str(x) for x in reason_codes])
        variant = s.get("variant")
        paraphrase_id = s.get("paraphrase_id")
        perturbation_kind = s.get("perturbation_kind")
        # Fallbacks from nested metadata if present
        meta = s.get("metadata", {}) or {}
        if variant is None:
            variant = meta.get("variant")
        if paraphrase_id is None:
            paraphrase_id = meta.get("paraphrase_id")
        if perturbation_kind is None:
            perturbation_kind = meta.get("perturbation_kind")

        results.append(
            SampleResult(
                id=sid,
                task=task,
                model=model,
                correct=correct,
                pred_index=pred_idx,
                target_index=target_idx,
                confidence=conf,
                seed=seed,
                flag_predictable=flag_predictable,
                predictability_score=predictability_score,
                probe_hit=probe_hit,
                ambiguity_label=ambiguity_label,
                reason_codes=reason_codes,
                variant=variant,
                paraphrase_id=paraphrase_id,
                perturbation_kind=perturbation_kind,
            )
        )
    return results


def load_all_results(logs_dir: str = "logs") -> pd.DataFrame:
    p = Path(logs_dir)
    if not p.exists():
        return pd.DataFrame()
    rows: list[SampleResult] = []
    for f in _collect_log_files(p):
        try:
            rows.extend(_parse_inspect_log(f))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


def majority_consensus(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    """Mark item exploitable if majority of models correct in choices-only."""
    if df.empty:
        return df
    mask = df["task"].fillna("").str.contains("choices_only")
    co = df[mask]
    if co.empty:
        return df
    grp = co.groupby(["id"])  # aggregate across models
    exploitable = grp["correct"].apply(lambda s: int(s.sum() >= k)).rename("exploitable")
    return df.merge(exploitable, left_on="id", right_index=True, how="left")


def mcq_cloze_gap(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"gap": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    # Compute per-item accuracy for mcq_full and cloze_full by model
    task_col = df["task"].fillna("")
    mcq = df[task_col.str.contains("mcq") & ~task_col.str.contains("choices_only")]
    cloze = df[task_col.str.contains("cloze")]
    if mcq.empty or cloze.empty:
        return {"gap": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    key = ["id", "model"]
    merged = mcq.merge(cloze, on=key, suffixes=("_mcq", "_cloze"))
    merged["delta"] = merged["correct_mcq"].astype(float) - merged["correct_cloze"].astype(float)
    vals = merged["delta"].tolist()
    mean = sum(vals) / len(vals) if vals else 0.0
    lo, hi = bootstrap_ci(vals) if vals else (0.0, 0.0)
    return {"gap": mean, "ci_lo": lo, "ci_hi": hi}


def abstention_overconfidence(df: pd.DataFrame) -> dict[str, float]:
    if df.empty:
        return {"abstention_rate": 0.0, "overconfidence_rate": 0.0}
    # Treat confidence==0 or missing prediction as abstention proxy if explicit flag missing
    abst_mask = (df["confidence"].fillna(-1) == 0) | (df["pred_index"].isna())
    abst_rate = float(abst_mask.mean())
    wrong = df["correct"] == False  # noqa: E712
    overconf = wrong & (~abst_mask)
    overconf_rate = float(overconf.mean())
    return {"abstention_rate": abst_rate, "overconfidence_rate": overconf_rate}


def longest_answer_heuristic(df: pd.DataFrame) -> dict[str, float]:
    # Heuristic accuracy if logs include choice lengths in metadata
    # If not available, return zeros gracefully
    return {"longest_answer_acc": 0.0}


def aggregate_main(logs_dir: str, out_dir: str) -> int:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    df = load_all_results(logs_dir)
    if df.empty:
        summary = {"note": "No logs found; run evals first."}
        Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=2))
        return 0
    df2 = majority_consensus(df)
    gap = mcq_cloze_gap(df2)
    abst = abstention_overconfidence(df2)

    summary = {
        "n_rows": int(df2.shape[0]),
        "models": sorted(df2["model"].dropna().unique().tolist()),
        "tasks": sorted(df2["task"].dropna().unique().tolist()),
        "mcq_vs_cloze": gap,
        "abstention_overconfidence": abst,
    }
    Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    df2.to_csv(Path(out_dir, "all_results.csv"), index=False)
    return 0


def cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate Inspect logs for RobustCBRN")
    ap.add_argument("--logs", default=get_paths().logs_dir)
    ap.add_argument("--out", default=get_paths().results_dir)
    args = ap.parse_args(argv)
    return aggregate_main(args.logs, args.out)


if __name__ == "__main__":
    raise SystemExit(cli())
