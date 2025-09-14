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
    # Pair bookkeeping (for benign policy pairs and similar)
    pair_id: Optional[str] = None


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
        pair_id = s.get("pair_id")
        # Fallbacks from nested metadata if present
        meta = s.get("metadata", {}) or {}
        if variant is None:
            variant = meta.get("variant")
        if paraphrase_id is None:
            paraphrase_id = meta.get("paraphrase_id")
        if perturbation_kind is None:
            perturbation_kind = meta.get("perturbation_kind")
        if pair_id is None:
            pair_id = meta.get("pair_id")
        # Fallback: infer pair_id from id prefix like "<rid>.<variant>"
        if (pair_id is None) and sid:
            if isinstance(sid, str) and "." in sid:
                pair_id = sid.split(".", 1)[0]

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
                pair_id=pair_id,
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
    # Coerce to numeric first to avoid pandas FutureWarning about downcasting on fillna
    conf_num = pd.to_numeric(df.get("confidence"), errors="coerce")
    abst_mask = (conf_num.fillna(-1.0) == 0.0) | (df.get("pred_index").isna())
    abst_rate = float(abst_mask.mean())
    wrong = df["correct"] == False  # noqa: E712
    overconf = wrong & (~abst_mask)
    overconf_rate = float(overconf.mean())
    return {"abstention_rate": abst_rate, "overconfidence_rate": overconf_rate}


def longest_answer_heuristic(df: pd.DataFrame) -> dict[str, float]:
    # Heuristic accuracy if logs include choice lengths in metadata
    # If not available, return zeros gracefully
    return {"longest_answer_acc": 0.0}


# Import moved inside function to avoid circular import


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

    # Benign pair stress metrics (if present)
    # Import here to avoid circular dependency
    from .robustness import benign_pair_metrics
    benign = benign_pair_metrics(df2)

    # Formal measurement: McNemar per task with variants (orig vs non-orig)
    from .robustness import (
        bh_fdr,
        consistency_at_k,
        delta_accuracy,
        fragility_score,
        mcnemar_orig_vs_variants,
        multi_reference_match_rates,
        power_two_proportions,
        required_sample_size_two_proportions,
    )

    tasks = sorted(df2["task"].dropna().unique().tolist())
    mcnemar_results: list[dict] = []
    pvals: list[float] = []
    for t in tasks:
        sub = df2[df2["task"] == t]
        if sub.empty or "variant" not in sub.columns:
            continue
        v = sub["variant"].fillna("")
        if not ((v == "orig").any() and (v != "orig").any()):
            continue
        res = mcnemar_orig_vs_variants(df2, task_name=t)
        res_task = {"task": t, **res}
        mcnemar_results.append(res_task)
        pvals.append(float(res.get("p_value", 1.0)))
    # Apply BH-FDR if we have multiple tests
    fdr = bh_fdr(pvals, alpha=0.05) if pvals else {"rejected": [], "q_values": [], "alpha": 0.05}
    for i, r in enumerate(mcnemar_results):
        r["q_value"] = float(fdr["q_values"][i]) if fdr["q_values"] else 1.0
        r["rejected"] = bool(fdr["rejected"][i]) if fdr["rejected"] else False

    # Multi-reference match rates if text columns are present
    multiref = multi_reference_match_rates(df2)

    # Power analysis for MCQ vs Cloze using observed accuracies
    # Reuse join logic from mcq_cloze_gap
    task_col = df2["task"].fillna("")
    mcq = df2[task_col.str.contains("mcq") & ~task_col.str.contains("choices_only")]
    cloze = df2[task_col.str.contains("cloze")]
    power_stats = {}
    if not mcq.empty and not cloze.empty:
        key = ["id", "model"]
        merged = mcq.merge(cloze, on=key, suffixes=("_mcq", "_cloze"))
        if not merged.empty:
            p1 = float(merged["correct_mcq"].astype(float).mean())
            p2 = float(merged["correct_cloze"].astype(float).mean())
            n_pairs = int(merged.shape[0])
            n1_req, n2_req = required_sample_size_two_proportions(p1, p2, alpha=0.05, power=0.8)
            pow_obs = power_two_proportions(p1, p2, n_pairs, n_pairs, alpha=0.05)
            power_stats = {
                "p1": p1,
                "p2": p2,
                "n_pairs": n_pairs,
                "observed_delta": float(abs(p1 - p2)),
                "observed_power": pow_obs,
                "required_n1": int(n1_req),
                "required_n2": int(n2_req),
            }

    # Robustness metrics: paraphrase consistency, perturbation fragility, delta accuracy
    try:
        paraphrase_consistency = consistency_at_k(df2, task_name="paraphrase_consistency")
    except Exception:
        paraphrase_consistency = {"n": 0, "consistency": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    try:
        perturbation_fragility = fragility_score(df2, task_name="perturbation_stability")
    except Exception:
        perturbation_fragility = {"n": 0, "flip_rate": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    try:
        paraphrase_delta_accuracy = delta_accuracy(df2, task_name="paraphrase_consistency")
    except Exception:
        paraphrase_delta_accuracy = {"n": 0, "delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}

    summary = {
        "n_rows": int(df2.shape[0]),
        "models": sorted(df2["model"].dropna().unique().tolist()),
        "tasks": sorted(df2["task"].dropna().unique().tolist()),
        "mcq_vs_cloze": gap,
        "abstention_overconfidence": abst,
        "benign_pair_stress": benign,
        "mcnemar": mcnemar_results,
        "fdr": {"alpha": float(fdr.get("alpha", 0.05))} if mcnemar_results else {},
        "multi_reference": multiref,
        "power_mcq_vs_cloze": power_stats,
        "paraphrase_consistency": paraphrase_consistency,
        "perturbation_fragility": perturbation_fragility,
        "paraphrase_delta_accuracy": paraphrase_delta_accuracy,
    }
    Path(out_dir, "summary.json").write_text(json.dumps(summary, indent=2))
    df2.to_csv(Path(out_dir, "all_results.csv"), index=False)

    # Generate example figures under results/figs
    try:
        from .figs import save_bar_ci, save_fragility, save_paired_delta
        # Save figures under configured figs_dir (e.g., artifacts/figs)
        figs_dir = Path(get_paths().figs_dir)
        figs_dir.mkdir(parents=True, exist_ok=True)
        # MCQ vs Cloze paired delta
        if gap and ("gap" in gap):
            save_paired_delta(
                figs_dir / "mcq_cloze_delta.png",
                labels=["All"],
                deltas=[float(gap.get("gap", 0.0))],
                ci_los=[float(gap.get("ci_lo", 0.0))],
                ci_his=[float(gap.get("ci_hi", 0.0))],
                title="MCQ − Cloze Δ (95% CI)",
                ylabel="Δ accuracy",
            )
        # Benign pairs summary bars (if present)
        if benign and benign.get("present", False):
            labels = [
                "Compliance (safe)",
                "Refusal (unsafe)",
                "Pair consistency",
            ]
            means = [
                float(benign.get("compliance_safe", 0.0)),
                float(benign.get("refusal_unsafe", 0.0)),
                float(benign.get("pair_consistency", 0.0)),
            ]
            ci_los = [
                float((benign.get("compliance_safe_ci", [0.0, 0.0]) or [0.0, 0.0])[0]),
                float((benign.get("refusal_unsafe_ci", [0.0, 0.0]) or [0.0, 0.0])[0]),
                float((benign.get("pair_consistency_ci", [0.0, 0.0]) or [0.0, 0.0])[0]),
            ]
            ci_his = [
                float((benign.get("compliance_safe_ci", [0.0, 0.0]) or [0.0, 0.0])[1]),
                float((benign.get("refusal_unsafe_ci", [0.0, 0.0]) or [0.0, 0.0])[1]),
                float((benign.get("pair_consistency_ci", [0.0, 0.0]) or [0.0, 0.0])[1]),
            ]
            save_bar_ci(
                figs_dir / "benign_pairs.png",
                labels=labels,
                means=means,
                ci_los=ci_los,
                ci_his=ci_his,
                title="Benign Policy Pairs (95% CI)",
                ylabel="Rate",
            )
        # Paraphrase consistency figure
        if paraphrase_consistency and int(paraphrase_consistency.get("n", 0)) > 0:
            save_bar_ci(
                figs_dir / "paraphrase_consistency.png",
                labels=["Consistency"],
                means=[float(paraphrase_consistency.get("consistency", 0.0))],
                ci_los=[float(paraphrase_consistency.get("ci_lo", 0.0))],
                ci_his=[float(paraphrase_consistency.get("ci_hi", 0.0))],
                title="Paraphrase Consistency (95% CI)",
                ylabel="Fraction",
            )
        # Perturbation fragility figure
        if perturbation_fragility and int(perturbation_fragility.get("n", 0)) > 0:
            save_fragility(
                figs_dir / "perturbation_fragility.png",
                labels=["Flip rate"],
                flip_rates=[float(perturbation_fragility.get("flip_rate", 0.0))],
                ci_los=[float(perturbation_fragility.get("ci_lo", 0.0))],
                ci_his=[float(perturbation_fragility.get("ci_hi", 0.0))],
                title="Perturbation Fragility (95% CI)",
                ylabel="Rate",
            )
    except Exception:
        # Figures are optional; ignore plotting errors in headless or minimal environments
        pass
    return 0


def cli(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate Inspect logs for RobustCBRN")
    ap.add_argument("--logs", default=get_paths().logs_dir)
    ap.add_argument("--out", default=get_paths().results_dir)
    args = ap.parse_args(argv)
    return aggregate_main(args.logs, args.out)


if __name__ == "__main__":
    raise SystemExit(cli())
