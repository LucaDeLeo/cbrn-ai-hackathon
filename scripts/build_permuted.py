#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random

try:  # allow both execution styles
    from .utils import read_jsonl
except Exception:  # pragma: no cover
    from utils import read_jsonl


def main():
    ap = argparse.ArgumentParser(description="Permute choices and update answer index")
    ap.add_argument("in_path")
    ap.add_argument("out_path")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    with open(args.out_path, "w", encoding="utf-8") as w:
        for doc in read_jsonl(args.in_path):
            n = len(doc["choices"])
            order = list(range(n))
            rng.shuffle(order)
            new_choices = [doc["choices"][i] for i in order]
            new_answer = order.index(doc["answer"])
            new_id = f'{doc.get("id", "noid")}_perm1'
            out = {
                **doc,
                "id": new_id,
                "choices": new_choices,
                "answer": new_answer,
            }
            w.write(json.dumps(out) + "\n")
    print(f"Wrote permuted dataset to {args.out_path}")


if __name__ == "__main__":
    main()

