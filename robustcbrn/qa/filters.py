from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


def _softmax(xs: Sequence[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps) or 1.0
    return [x / s for x in exps]


@dataclass
class Item:
    id: str
    choices: list[str]
    target: int


@dataclass
class PredictabilityResult:
    id: str
    predictability_score: float
    flag_predictable: bool
    probe_hit: list[str]


def _as_items(dataset: Iterable) -> list[Item]:
    items: list[Item] = []
    for s in dataset:
        if hasattr(s, "choices"):
            sid = getattr(s, "id", None)
            choices = list(s.choices)
            target = int(s.target)
        else:
            sid = s.get("id")  # type: ignore[index]
            choices = list(s.get("choices", []))  # type: ignore[index]
            target = int(s.get("target"))  # type: ignore[index]
        items.append(Item(id=str(sid), choices=choices, target=target))
    return items


def _build_vocab(items: list[Item], max_features: int = 2000) -> dict[str, int]:
    df: Counter[str] = Counter()
    for it in items:
        for ch in it.choices:
            toks = set(_tokenize(ch))
            df.update(toks)
    most = [w for w, _ in df.most_common(max_features)]
    return {w: i for i, w in enumerate(most)}


def _featurize_candidates(
    items: list[Item], vocab: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Create per-candidate features for binary logistic regression.

    Returns X, y, item_ids, pos_indices where each row corresponds to one choice.
    """
    rows: list[list[float]] = []
    labels: list[int] = []
    item_ids: list[str] = []
    pos_indices: list[int] = []  # within-item choice index
    v = len(vocab)
    for it in items:
        n = len(it.choices)
        # Precompute within-item stats
        char_lens = [len(c) for c in it.choices]
        token_lens = [len(_tokenize(c)) for c in it.choices]
        longest_idx = int(max(range(n), key=lambda i: char_lens[i])) if n > 0 else 0
        # Rank lengths descending (ties keep lower index)
        ranks = np.argsort([-L for L in char_lens]) if n > 0 else np.array([0])
        inv_rank = {int(ranks[i]): i for i in range(len(ranks))}
        max_char = max(char_lens) if char_lens else 1
        max_tok = max(token_lens) if token_lens else 1

        for j, choice in enumerate(it.choices):
            bow = [0.0] * v
            for t in _tokenize(choice):
                idx = vocab.get(t)
                if idx is not None:
                    bow[idx] += 1.0
            c_len = float(char_lens[j])
            t_len = float(token_lens[j])
            avg_w = (c_len / max(1.0, t_len)) if t_len > 0 else c_len
            pos_feat = j / max(1.0, n - 1) if n > 1 else 0.0
            is_longest = 1.0 if j == longest_idx else 0.0
            rel_len_rank = float(inv_rank.get(j, 0)) / max(1.0, n - 1) if n > 1 else 0.0
            first_char = (it.choices[j].strip()[:1] or " ").lower()
            first_ord = (
                (ord(first_char) - ord("a")) / 25.0 if first_char.isalpha() else 1.0
            )

            # Normalize counts by length
            if c_len > 0:
                bow = [b / c_len for b in bow]

            extra = [
                c_len / max(1.0, max_char),
                t_len / max(1.0, max_tok),
                avg_w / max(1.0, (max_char / max(1.0, max_tok))),
                pos_feat,
                is_longest,
                rel_len_rank,
                first_ord,
            ]
            rows.append(bow + extra)
            labels.append(1 if j == it.target else 0)
            item_ids.append(it.id)
            pos_indices.append(j)

    x = np.asarray(rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    return x, y, item_ids, pos_indices


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logreg_l2(
    x: np.ndarray, y: np.ndarray, reg_lambda: float = 1.0, maxiter: int = 200
) -> np.ndarray:
    """Train binary logistic regression with L2 using SciPy if available.

    Returns weight vector including bias term (shape D+1,).
    """
    xb = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    d = xb.shape[1]
    w0 = np.zeros(d, dtype=x.dtype)

    try:
        from scipy.optimize import minimize  # type: ignore

        def obj(w: np.ndarray) -> tuple[float, np.ndarray]:
            z = xb @ w
            # Stable log-loss: log(1+exp(z)) - y*z
            # Use logaddexp to improve stability
            loss_vec = np.logaddexp(0.0, z) - y * z
            # L2 reg (exclude bias)
            reg = 0.5 * reg_lambda * np.sum(w[:-1] * w[:-1])
            loss = float(np.sum(loss_vec) + reg)
            # Gradient
            p = _sigmoid(z)
            grad = xb.T @ (p - y)
            grad[:-1] += reg_lambda * w[:-1]
            return loss, grad.astype(np.float64)

        res = minimize(
            fun=lambda w: obj(w)[0],
            x0=w0,
            jac=lambda w: obj(w)[1],
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-8},
        )
        w = res.x.astype(x.dtype)
        return w
    except Exception:
        # Simple fallback: gradient descent
        w = w0
        lr = 0.1
        for _ in range(maxiter):
            z = xb @ w
            p = _sigmoid(z)
            grad = xb.T @ (p - y)
            grad[:-1] += reg_lambda * w[:-1]
            w = w - lr * grad / max(1.0, xb.shape[0])
        return w


def _kfold_by_item(item_ids: list[str], k: int, seed: int = 123) -> list[set[int]]:
    rng = np.random.RandomState(seed)
    unique = sorted(set(item_ids))
    rng.shuffle(unique)
    folds: list[set[str]] = [set() for _ in range(max(2, k))]
    for i, iid in enumerate(unique):
        folds[i % len(folds)].add(iid)  # type: ignore[arg-type]
    index_folds: list[set[int]] = []
    for fold_items in folds:
        idxs = {i for i, iid in enumerate(item_ids) if iid in fold_items}
        index_folds.append(idxs)
    return index_folds


def _alphabetical_choice(choices: list[str]) -> int:
    if not choices:
        return 0
    return int(min(range(len(choices)), key=lambda i: choices[i].strip().lower()))


def _longest_choice(choices: list[str]) -> int:
    if not choices:
        return 0
    return int(max(range(len(choices)), key=lambda i: len(choices[i])))


def _position_majority_train(items: list[Item]) -> int:
    counts = Counter(int(it.target) for it in items)
    if not counts:
        return 0
    # Majority index; ties broken by lower index
    maj = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return int(maj)


def aflite_lite(
    dataset: Iterable,
    tau: float = 0.7,
    k_folds: int = 5,
    seed: int = 123,
    max_vocab: int = 2000,
    reg_lambda: float = 1.0,
) -> list[PredictabilityResult]:
    """Compute AFLite-lite predictability and heuristic probe hits.

    - Trains a choices-only per-candidate logistic regression with K-fold OOF.
    - Emits per-item predictability score (probability correct for true choice),
      flag_predictable using threshold tau, and probe_hit list for heuristics
      that correctly predict the item (longest-answer, position-only, alphabetical).
    """
    items = _as_items(dataset)
    if not items:
        return []
    vocab = _build_vocab(items, max_features=max_vocab)
    x, y, item_ids, pos_indices = _featurize_candidates(items, vocab)

    # Build per-fold train/test based on item id groups (avoid leakage)
    folds = _kfold_by_item([it.id for it in items for _ in it.choices], k=k_folds, seed=seed)

    # Precompute mapping from candidate row index to item index and choice index
    row_to_item_idx: list[int] = []
    row = 0
    for i, it in enumerate(items):
        for _ in it.choices:
            row_to_item_idx.append(i)
            row += 1

    # Collect per-candidate probabilities via OOF
    probs = np.zeros_like(y)

    for _fi, test_idx_set in enumerate(folds):
        if not test_idx_set:
            continue
        train_mask = np.ones(len(y), dtype=bool)
        test_mask = np.zeros(len(y), dtype=bool)
        # Mark rows belonging to fold item ids
        for ridx in range(len(y)):
            row_to_item_idx[ridx]
            # Determine if this row belongs in the current test fold
            if ridx in test_idx_set:
                test_mask[ridx] = True
                train_mask[ridx] = False
        xtr, ytr = x[train_mask], y[train_mask]
        xte = x[test_mask]
        if xtr.shape[0] == 0:
            continue
        w = _fit_logreg_l2(xtr, ytr, reg_lambda=reg_lambda)
        xteb = np.concatenate([xte, np.ones((xte.shape[0], 1), dtype=xte.dtype)], axis=1)
        z = xteb @ w
        p = _sigmoid(z)
        probs[test_mask] = p

    # Aggregate per-item statistics
    results: list[PredictabilityResult] = []
    # Build fold-specific position-only predictor: use global (all) majority as approximation
    pos_majority_idx = _position_majority_train(items)

    offset = 0
    for it in items:
        n = len(it.choices)
        cand_probs = probs[offset : offset + n].tolist()
        offset += n

        # Probability that the true choice is correct (our score)
        score = float(cand_probs[it.target]) if 0 <= it.target < n else 0.0
        hits: list[str] = []
        # Heuristic probes
        if _longest_choice(it.choices) == it.target:
            hits.append("longest_answer")
        if pos_majority_idx < n and pos_majority_idx == it.target:
            hits.append("position_only")
        if _alphabetical_choice(it.choices) == it.target:
            hits.append("alphabetical")

        results.append(
            PredictabilityResult(
                id=it.id,
                predictability_score=score,
                flag_predictable=bool(score >= tau),
                probe_hit=hits,
            )
        )
    return results


def aflite_join_with_items(
    items: Iterable, results: list[PredictabilityResult]
) -> list[dict]:
    """Join AFLite-lite outputs back onto item dicts for logging or export."""
    by_id = {r.id: r for r in results}
    joined: list[dict] = []
    for s in items:
        if hasattr(s, "choices"):
            sid = getattr(s, "id", None)
            choices = list(s.choices)
            target = int(s.target)
            rec = {
                "id": str(sid),
                "choices": choices,
                "target": target,
            }
        else:
            rec = {
                "id": str(s.get("id")),  # type: ignore[index]
                "choices": list(s.get("choices", [])),  # type: ignore[index]
                "target": int(s.get("target")),  # type: ignore[index]
            }
        r = by_id.get(str(rec["id"]))
        if r is not None:
            rec.update(
                {
                    "predictability_score": r.predictability_score,
                    "flag_predictable": r.flag_predictable,
                    "probe_hit": r.probe_hit,
                }
            )
        joined.append(rec)
    return joined

