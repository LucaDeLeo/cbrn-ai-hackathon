from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List


def hash_id(s: str, salt: str = "robustcbrn") -> str:
    return hashlib.sha256((salt + "::" + s).encode()).hexdigest()[:16]


def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def try_get(doc: Dict, *keys, default=None):
    for k in keys:
        if k in doc:
            return doc[k]
    return default


def normalize_prediction(sample: Dict) -> Dict:
    """
    Return a dict with fields:
      id, choices (list[str]), target_idx, pred_idx (may be None),
      choice_logprobs (list[float]) if present
    Handles various lm-eval sample formats.
    """
    doc = sample.get("doc") or {}
    sid = try_get(sample, "doc_id", "id", default=doc.get("id", ""))
    choices = try_get(
        doc,
        "choices",
        default=try_get(sample, "choices", default=[]),
    )
    # target
    tgt_idx = None
    if "answer" in doc and isinstance(doc["answer"], int):
        tgt_idx = doc["answer"]
    elif "target" in sample and choices:
        # if target is string, map to index
        t = sample["target"]
        if isinstance(t, str) and t in choices:
            tgt_idx = choices.index(t)
    # prediction
    pred_idx = try_get(sample, "choice_index", "pred_idx", default=None)
    pred = sample.get("prediction")
    if pred_idx is None and pred is not None:
        if isinstance(pred, int):
            pred_idx = pred
        elif isinstance(pred, str) and choices and pred in choices:
            pred_idx = choices.index(pred)
        elif isinstance(pred, str) and pred in "ABCD" and len(choices) == 4:
            pred_idx = "ABCD".index(pred)
    # choice logprobs / scores
    clp = try_get(sample, "choice_logprob", "choice_logprobs", default=None)
    if clp is None:
        scores = try_get(sample, "choice_scores", default=None)
        if scores and all(isinstance(x, (int, float)) for x in scores):
            # interpret as logprob-like scores
            clp = scores
    return {
        "id": sid,
        "choices": choices,
        "target_idx": tgt_idx,
        "pred_idx": pred_idx,
        "choice_logprobs": clp,
    }

