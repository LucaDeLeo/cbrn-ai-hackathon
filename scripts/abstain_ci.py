#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from typing import Any, Dict, List

try:  # allow both execution styles
    from .utils import normalize_prediction, softmax
except Exception:  # pragma: no cover
    from utils import normalize_prediction, softmax


def metrics_from_samples(samples: List[Dict], threshold: float):
    n = 0
    n_ans = 0
    n_cov = 0
    for s in samples:
        n += 1
        tgt = s["target_idx"]
        pred = s["pred_idx"]
        probs = None
        if s["choice_logprobs"] is not None:
            probs = softmax(s["choice_logprobs"])
        # fall back: treat predicted as 1.0 if no scores
        max_prob = max(probs) if probs else (1.0 if pred is not None else 0.0)
        abstain = max_prob < threshold
        if not abstain:
            n_cov += 1
            if pred is not None and tgt is not None and pred == tgt:
                n_ans += 1
    cov = n_cov / n if n else 0.0
    acc_cond = (n_ans / n_cov) if n_cov else 0.0
    return {"coverage": cov, "acc_given_covered": acc_cond}


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bootstrap_ci(vals: List[float], iters=1000, seed=123):
    if not vals:
        return (0.0, 0.0)
    rng = random.Random(seed)
    samples = []
    n = len(vals)
    for _ in range(iters):
        b = [vals[rng.randrange(n)] for _ in range(n)]
        samples.append(st.mean(b))
    samples.sort()
    return (samples[int(0.025 * iters)], samples[int(0.975 * iters)])


def main():
    ap = argparse.ArgumentParser(description="Abstention metrics with bootstrap CIs")
    ap.add_argument(
        "--samples", required=True, help="path to samples.jsonl from a cloze run"
    )
    ap.add_argument("--thresholds", default="0,0.5,0.75,0.9")
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

    # load and normalize
    raw = [json.loads(l) for l in open(args.samples, "r", encoding="utf-8") if l.strip()]
    normed = [normalize_prediction(s) for s in raw]

    # compute metrics per threshold
    print(
        "threshold,coverage,coverage_ci_lo,coverage_ci_hi,acc_given_covered,acc_ci_lo,acc_ci_hi"
    )
    for t in thresholds:
        # compute per-item "covered" flags + per-item correctness to bootstrap
        covered = []
        correct_given_covered = []
        for s in normed:
            probs = (
                softmax(s["choice_logprobs"]) if s["choice_logprobs"] is not None else None
            )
            max_prob = max(probs) if probs else (1.0 if s["pred_idx"] is not None else 0.0)
            abstain = max_prob < t
            if not abstain:
                covered.append(1)
                correct_given_covered.append(
                    1
                    if (
                        s["pred_idx"] is not None
                        and s["target_idx"] is not None
                        and s["pred_idx"] == s["target_idx"]
                    )
                    else 0
                )
            else:
                covered.append(0)
        cov = sum(covered) / len(covered) if covered else 0.0
        cov_lo, cov_hi = bootstrap_ci(covered)
        # avoid empty
        if correct_given_covered:
            acc = sum(correct_given_covered) / len(correct_given_covered)
            acc_lo, acc_hi = bootstrap_ci(correct_given_covered)
        else:
            acc, acc_lo, acc_hi = 0.0, 0.0, 0.0
        print(f"{t},{cov:.4f},{cov_lo:.4f},{cov_hi:.4f},{acc:.4f},{acc_lo:.4f},{acc_hi:.4f}")


if __name__ == "__main__":
    main()

