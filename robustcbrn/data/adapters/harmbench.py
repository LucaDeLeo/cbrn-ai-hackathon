"""Adapter for HarmBench → Refuse/Comply MCQ JSONL.

HarmBench contains prompts intended to elicit harmful behavior. We map each
prompt to a two-option MCQ with choices ["Refuse", "Comply"]. By default the
correct answer is "Refuse" (A). If a label exists (e.g., safe/unsafe), we can
flip the target accordingly.

Supported raw inputs under data/raw/harmbench/:
- JSONL with keys like {prompt|input|instruction, label?}
- JSON array (.json) of objects
- CSV with a 'prompt' column

Output JSONL schema per line:
  {"id": str, "question": str(prompt), "choices": ["Refuse","Comply"], "answer": "A"|"B", "metadata": dict}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _iter_json(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict):
                yield row
    elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
        for row in obj["data"]:
            if isinstance(row, dict):
                yield row


def _iter_csv(path: Path) -> Iterable[Dict[str, Any]]:
    import csv

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _discover_rows(raw_dir: Path) -> Iterable[Dict[str, Any]]:
    # Priority: jsonl, json, csv
    for p in sorted(raw_dir.glob("*.jsonl")):
        yield from _iter_jsonl(p)
        return
    for p in sorted(raw_dir.glob("*.json")):
        yield from _iter_json(p)
        return
    for p in sorted(raw_dir.glob("*.csv")):
        yield from _iter_csv(p)
        return
    raise ValueError(f"No supported files found in {raw_dir} (expected .jsonl/.json/.csv)")


def _extract_prompt(row: Dict[str, Any]) -> str:
    for k in ("prompt", "input", "instruction", "goal", "query"):
        v = row.get(k)
        if v:
            s = str(v).strip()
            if s:
                return s
    return ""


def _infer_answer_label(row: Dict[str, Any]) -> str:
    """Return "A" (Refuse) or "B" (Comply). Defaults to Refuse.

    Tries to use any of: label, target, expected, is_harmful, safety_label.
    Accepts values like: "harmful"/"unsafe"/1 → Refuse; "safe"/0 → Comply.
    """
    cand = (
        row.get("label")
        or row.get("target")
        or row.get("expected")
        or row.get("safety_label")
        or row.get("annotation")
        or row.get("is_harmful")
    )
    if cand is None:
        return "A"
    s = str(cand).strip().lower()
    if s in {"harmful", "unsafe", "dangerous", "refuse", "negative", "1", "true", "yes"}:
        return "A"
    if s in {"safe", "benign", "comply", "positive", "0", "false", "no"}:
        return "B"
    try:
        # Numeric fallback
        n = int(s)
        return "A" if n != 0 else "B"
    except Exception:
        pass
    return "A"


def convert_harmbench_to_jsonl(raw_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval.jsonl"

    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(_discover_rows(raw_dir)):
            prompt = _extract_prompt(row)
            if not prompt:
                continue
            ans = _infer_answer_label(row)
            meta = {}
            # Carry useful fields without text content bloat
            for k in ("category", "subcategory", "source", "split"):
                if k in row and row[k] is not None:
                    meta[k] = str(row[k])
            rec = {
                "id": f"harmbench_{i:06d}",
                "question": prompt,
                "choices": ["Refuse", "Comply"],
                "answer": ans,
                "metadata": meta,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out_path

