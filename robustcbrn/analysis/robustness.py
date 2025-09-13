from __future__ import annotations

from typing import Optional

import pandas as pd

from .aggregate import majority_consensus
from ..utils.stats import bootstrap_ci


def _to_list(val) -> list[str]:
    if val is None:
        return []
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        # comma or semicolon separated
        parts = [x.strip() for x in val.split(",") if x.strip()]
        return parts
    return [str(val)]


def _apply_preset(
    df_flags: pd.DataFrame, preset: str = "balanced", tau: float = 0.7
) -> pd.Series:
    score = df_flags["predictability_score"].astype(float).fillna(0.0)
    hits = df_flags["probe_hit"].apply(_to_list)
    if preset == "conservative":
        mask = (score >= max(tau, 0.8)) & (hits.apply(lambda xs: len(xs) >= 2))
    elif preset == "aggressive":
        mask = (score >= min(tau, 0.6)) | (hits.apply(lambda xs: len(xs) >= 1))
    else:  # balanced
        mask = score >= tau
    return mask.astype(bool).rename("flag_predictable_preset")


def aflite_flag_summary(
    df: pd.DataFrame, preset: str = "balanced", tau: float = 0.7
) -> dict[str, float]:
    if df.empty:
        return {"n": 0, "flagged_frac": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    mask = (df.get("task", "") == "aflite_screen") | df.get("predictability_score").notna()
    sub = df[mask].copy()
    if sub.empty:
        return {"n": 0, "flagged_frac": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    # Collapse to one row per id
    grp = sub.groupby("id")
    agg = grp.agg(
        predictability_score=("predictability_score", "max"),
        probe_hit=("probe_hit", "first"),
    ).reset_index()
    flags = _apply_preset(agg, preset=preset, tau=tau)
    vals = flags.astype(int).tolist()
    frac = float(sum(vals) / len(vals)) if vals else 0.0
    lo, hi = bootstrap_ci([float(v) for v in vals]) if vals else (0.0, 0.0)
    return {"n": int(len(vals)), "flagged_frac": frac, "ci_lo": lo, "ci_hi": hi}


def aflite_overlap_with_choices_only(
    df: pd.DataFrame, preset: str = "balanced", tau: float = 0.7, k: int = 2
) -> dict[str, float]:
    if df.empty:
        return {
            "n": 0,
            "flagged_frac": 0.0,
            "exploitable_frac": 0.0,
            "overlap_frac": 0.0,
            "jaccard": 0.0,
        }
    df2 = majority_consensus(df, k=k)
    flags = df2[(df2.get("task", "") == "aflite_screen") | df2.get("predictability_score").notna()]
    if flags.empty:
        return {
            "n": 0,
            "flagged_frac": 0.0,
            "exploitable_frac": 0.0,
            "overlap_frac": 0.0,
            "jaccard": 0.0,
        }
    # Collapse flags by id
    fa = (
        flags.groupby("id")
        .agg(
            predictability_score=("predictability_score", "max"),
            probe_hit=("probe_hit", "first"),
        )
        .reset_index()
    )
    fa["flag"] = _apply_preset(fa, preset=preset, tau=tau).astype(int)
    # Get exploitable by id
    ex = (
        df2.groupby("id")["exploitable"].max().fillna(0).astype(int).rename("exploitable").reset_index()
    )
    m = fa.merge(ex, on="id", how="outer").fillna(0)
    n = int(m.shape[0])
    if n == 0:
        return {
            "n": 0,
            "flagged_frac": 0.0,
            "exploitable_frac": 0.0,
            "overlap_frac": 0.0,
            "jaccard": 0.0,
        }
    flagged_frac = float(m["flag"].mean())
    exploitable_frac = float(m["exploitable"].mean())
    overlap = ((m["flag"] == 1) & (m["exploitable"] == 1)).mean()
    # Jaccard of sets (flagged U exploitable may be 0)
    inter = int(((m["flag"] == 1) & (m["exploitable"] == 1)).sum())
    union = int(((m["flag"] == 1) | (m["exploitable"] == 1)).sum())
    jaccard = float(inter / union) if union > 0 else 0.0
    return {
        "n": n,
        "flagged_frac": float(flagged_frac),
        "exploitable_frac": float(exploitable_frac),
        "overlap_frac": float(overlap),
        "jaccard": float(jaccard),
    }

