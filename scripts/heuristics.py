#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

try:  # allow both execution styles
    from .utils import read_jsonl
except Exception:  # pragma: no cover
    from utils import read_jsonl


def longest_answer_idx(choices):
    return max(range(len(choices)), key=lambda i: len(choices[i]))


def position_first_idx(choices):
    return 0


def position_last_idx(choices):
    return len(choices) - 1


def main():
    ap = argparse.ArgumentParser(description="Heuristic baselines on dataset JSONL")
    ap.add_argument("data", help="path to JSONL dataset")
    args = ap.parse_args()

    total = 0
    correct_longest = correct_first = correct_last = 0

    for doc in read_jsonl(args.data):
        total += 1
        ans = int(doc["answer"])
        ch = doc["choices"]
        if longest_answer_idx(ch) == ans:
            correct_longest += 1
        if position_first_idx(ch) == ans:
            correct_first += 1
        if position_last_idx(ch) == ans:
            correct_last += 1

    pct = lambda n: round(100 * n / total, 2)
    print(f"Samples: {total}")
    print(f"Longest-answer: {pct(correct_longest)}%")
    print(f"Position-first: {pct(correct_first)}%")
    print(f"Position-last: {pct(correct_last)}%")


if __name__ == "__main__":
    main()

