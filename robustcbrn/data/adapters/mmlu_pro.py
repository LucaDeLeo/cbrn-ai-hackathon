"""Adapter for TIGER-Lab/MMLU-Pro → RobustCBRN JSONL schema.

The adapter is intentionally tolerant to common HF layouts:
- JSONL with keys like {question, options|choices, answer|answer_index}
- Parquet with similar columns
- CSV with columns A/B/C/D… or choices/options

Output JSONL schema per line:
  {"id": str, "question": str, "choices": list[str], "answer": str(letter), "metadata": dict}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _to_choices(row: Dict[str, Any]) -> List[str]:
    # Prefer explicit list field
    for key in ("options", "choices", "options_labels"):
        if key in row and row[key] is not None:
            val = row[key]
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            # JSON-encoded list
            if isinstance(val, str) and val.strip().startswith("["):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
    # MMLU-style columns A, B, C, D, …
    opts: List[str] = []
    for letter in ["A", "B", "C", "D", "E", "F"]:
        if letter in row and row[letter]:
            opts.append(str(row[letter]).strip())
    if opts:
        return opts
    raise ValueError("Could not extract choices/options from row")


def _norm_answer(row: Dict[str, Any], n_choices: int) -> str:
    # Prefer explicit answer fields
    cand = row.get("answer")
    if cand is None:
        cand = row.get("answer_index")
    if cand is None:
        cand = row.get("label")
    if cand is None:
        # Some variants: 'target' or 'target_index'
        cand = row.get("target") if isinstance(row.get("target"), (int, str)) else row.get("target_index")
    if cand is None:
        # As a last resort, default to first option (A) — better than failing the whole conversion
        return "A"
    s = str(cand).strip()
    # Already a letter
    if s.upper() in ["A", "B", "C", "D", "E", "F"]:
        return s.upper()
    # Numeric index (0 or 1-based)
    try:
        idx = int(s)
        if 0 <= idx < n_choices:
            return chr(65 + idx)
        if 1 <= idx <= n_choices:
            return chr(65 + idx - 1)
    except Exception:
        pass
    # Try mapping exact text match to an option
    options = row.get("options") or row.get("choices")
    if isinstance(options, list):
        try:
            idx = [str(x).strip() for x in options].index(s)
            return chr(65 + idx)
        except Exception:
            pass
    # Fallback
    return "A"


def _iter_parquet(path: Path):
    import pandas as pd  # type: ignore

    df = pd.read_parquet(path)
    for _, r in df.iterrows():
        yield {k: r[k] for k in df.columns}


def _iter_csv(path: Path):
    import csv

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSONL at {path}: {e}") from e


def _discover_rows(raw_dir: Path):
    # Priority: jsonl, parquet, csv
    jsonls = sorted(raw_dir.glob("*.jsonl"))
    if jsonls:
        for p in jsonls:
            yield from _iter_jsonl(p)
        return
    pars = sorted(raw_dir.glob("*.parquet"))
    if pars:
        for p in pars:
            yield from _iter_parquet(p)
        return
    csvs = sorted(raw_dir.glob("*.csv"))
    if csvs:
        for p in csvs:
            yield from _iter_csv(p)
        return
    raise ValueError(f"No supported files found in {raw_dir} (expected .jsonl/.parquet/.csv)")


def convert_mmlu_pro_to_jsonl(raw_dir: Path, out_dir: Path) -> Path:
    """Convert MMLU-Pro raw files into eval.jsonl with RobustCBRN schema."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval.jsonl"

    with open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(_discover_rows(raw_dir)):
            q = str(row.get("question", row.get("prompt", row.get("input", "")))).strip()
            if not q:
                # Skip malformed row
                continue
            choices = _to_choices(row)
            if not choices:
                continue
            ans = _norm_answer(row, len(choices))
            # Metadata: include subject/category if present
            meta = {}
            for k in ("subject", "category", "subfield", "split", "source"):
                if k in row and row[k] is not None:
                    meta[k] = str(row[k])
            rec = {
                "id": f"mmlu_pro_{i:06d}",
                "question": q,
                "choices": choices,
                "answer": ans,
                "metadata": meta,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return out_path

