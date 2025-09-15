#!/usr/bin/env python3
import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any
import urllib.request

LETTER_RE = re.compile(r"([A-Z])")


def read_jsonl(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            items.append(obj)
            if limit is not None and len(items) >= limit:
                break
    return items


def to_index(answer: Any, num_choices: int) -> int | None:
    if answer is None:
        return None
    if isinstance(answer, int):
        return answer if 0 <= answer < num_choices else None
    if isinstance(answer, str) and answer:
        ch = answer.strip().upper()[0]
        if "A" <= ch <= "Z":
            return ord(ch) - ord("A")
        try:
            val = int(answer)
            return val if 0 <= val < num_choices else None
        except Exception:
            return None
    return None


def build_prompt(task: str, question: str | None, choices: List[str]) -> str:
    letters = [chr(ord('A') + i) for i in range(len(choices))]
    choices_str = "\n".join([f"{L}) {c}" for L, c in zip(letters, choices)])
    instr = (
        "You are a careful evaluator. Choose the single best option. "
        "Reply with only the letter (A, B, C, ...)."
    )
    if task == "mcq_choices_only":
        return f"{instr}\n\nChoices:\n{choices_str}\n\nAnswer:"
    else:
        stem = question or ""
        return f"{instr}\n\nQuestion: {stem}\n\nChoices:\n{choices_str}\n\nAnswer:"


def chat_complete(base_url: str, api_key: str, model: str, prompt: str) -> str:
    url = base_url.rstrip('/') + '/chat/completions'
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4,
        "n": 1
    }).encode('utf-8')
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    try:
        return data["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def parse_letter(text: str, num_choices: int) -> int | None:
    m = LETTER_RE.search(text.upper())
    if not m:
        return None
    idx = ord(m.group(1)) - ord('A')
    return idx if 0 <= idx < num_choices else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_path", required=True)
    ap.add_argument("--task", choices=["mcq_full", "mcq_choices_only", "cloze_full"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base_url", default=os.getenv("OPENAI_API_BASE", "http://localhost:8002/v1"))
    ap.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--logs_dir", default="logs")
    args = ap.parse_args()

    ds = read_jsonl(args.dataset_path, args.max_items)
    out_samples: List[Dict[str, Any]] = []

    for row in ds:
        sid = row.get("id")
        choices = row.get("choices") or []
        if not isinstance(choices, list) or len(choices) < 2:
            continue
        q = row.get("question") if args.task != "mcq_choices_only" else None
        prompt = build_prompt(args.task, q, [str(c) for c in choices])
        try:
            content = chat_complete(args.base_url, args.api_key, args.model, prompt)
        except Exception:
            content = ""
        pred = parse_letter(content, len(choices))
        target = to_index(row.get("answer"), len(choices))
        correct = (pred == target) if (pred is not None and target is not None) else None
        out_samples.append({
            "id": str(sid),
            "pred_index": pred,
            "target_index": target,
            "correct": correct,
            "num_choices": len(choices),
            "choice_lengths": [len(str(c)) for c in choices],
        })

    Path(args.logs_dir).mkdir(parents=True, exist_ok=True)
    model_sanitized = args.model.replace('/', '_')
    out = {
        "task": args.task,
        "model": args.model,
        "provider_model": args.model,
        "seed": args.seed,
        "samples": out_samples,
    }
    out_path = Path(args.logs_dir) / f"{args.task}_{model_sanitized}_{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"Wrote {out_path} ({len(out_samples)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

