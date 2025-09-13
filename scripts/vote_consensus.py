#!/usr/bin/env python
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

try:  # allow both `python -m scripts...` and `python scripts/...`
    from .utils import normalize_prediction, hash_id
except Exception:  # pragma: no cover
    from utils import normalize_prediction, hash_id


def main():
    ap = argparse.ArgumentParser(description="Majority vote over choices-only runs")
    ap.add_argument(
        "--glob",
        default="results/*/choicesonly/*/samples.jsonl",
        help="glob to samples.jsonl files from different models",
    )
    ap.add_argument("--out", default="results/consensus_choicesonly.csv")
    args = ap.parse_args()

    votes = defaultdict(list)

    for path in glob.glob(args.glob):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                norm = normalize_prediction(s)
                qid = hash_id(str(norm["id"]))
                votes[qid].append(norm["pred_idx"])

    rows = []
    for qid, preds in votes.items():
        cnt = Counter([p for p in preds if p is not None])
        top = cnt.most_common(1)[0] if cnt else (None, 0)
        majority = 1 if top[1] >= 2 else 0  # 2-of-3 majority
        rows.append(
            {
                "qid_hash": qid,
                "n_votes": len(preds),
                "maj_pred": top[0],
                "maj_count": top[1],
                "flag_shortcut_suspect": majority,
            }
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import csv

    with open(args.out, "w", newline="", encoding="utf-8") as w:
        writer = csv.DictWriter(w, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} with {len(rows)} rows")


if __name__ == "__main__":
    main()

