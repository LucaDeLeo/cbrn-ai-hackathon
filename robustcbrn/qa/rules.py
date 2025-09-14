from __future__ import annotations

"""
Automated QA & label hygiene checks.

This module performs lightweight, CPU-only validation of MCQ-style datasets:
- Near-duplicate detection via 64-bit SimHash over normalized text
- Label sanity checks (answer ∈ choices, index bounds)
- Choice duplication checks
- Simple grammar/lint heuristics (unbalanced quotes/brackets, repeated punctuation, etc.)

Outputs a public-safe CSV report with per-item flags and summary stats. No raw
stems or choice text are emitted.

Performance Limits:
- Optimal for datasets < 10,000 items
- O(n²) complexity for duplicate detection
- Memory usage: ~1KB per item
- Processing speed: ~1000 items/second on modern CPU
"""

import argparse
import csv
import hashlib
import json
import logging
import re
import signal
import sys
import time
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..utils.io import read_jsonl

# Try to import tqdm for progress bars, fall back to basic progress
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Constants
MAX_DATASET_SIZE = 50000  # Warn if dataset exceeds this
DEFAULT_REGEX_TIMEOUT = 1.0  # seconds
MAX_INPUT_SIZE = 10 * 1024 * 1024  # 10MB max file size


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+", re.UNICODE)


class RegexTimeoutError(Exception):
    """Raised when regex operation times out."""
    pass


@contextmanager
def timeout_handler(seconds: float = DEFAULT_REGEX_TIMEOUT):
    """Context manager for timing out operations."""
    def timeout_handler(signum, frame):
        raise RegexTimeoutError(f"Operation timed out after {seconds} seconds")

    if sys.platform != "win32":  # Unix-like systems
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(seconds + 1))
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        # Windows doesn't support SIGALRM, so we skip timeout
        yield


def validate_record(record: dict[str, Any]) -> tuple[bool, str | None]:
    """Validate a single record's schema.

    Returns:
        (is_valid, error_message)
    """
    if not isinstance(record, dict):
        return False, "Record must be a dictionary"

    # Check required fields
    if "id" not in record:
        return False, "Missing required field: id"

    # Check for question/input field
    if "question" not in record and "input" not in record:
        return False, "Missing required field: question or input"

    # Check choices field
    if "choices" not in record:
        return False, "Missing required field: choices"

    if not isinstance(record["choices"], list):
        return False, "Field 'choices' must be a list"

    if len(record["choices"]) == 0:
        return False, "Field 'choices' cannot be empty"

    # Check answer field
    if "answer" not in record:
        return False, "Missing required field: answer"

    # Check for excessively large fields
    for field in ["question", "input"]:
        if field in record and len(str(record[field])) > 100000:
            return False, f"Field '{field}' is too large (>100KB)"

    return True, None


def progress_bar(iterable: Iterable, total: int | None = None, desc: str = "Processing"):
    """Create a progress bar, using tqdm if available."""
    if HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, ncols=80)
    else:
        # Basic progress indicator
        items = list(iterable)
        total = len(items)
        for i, item in enumerate(items):
            if i % max(1, total // 20) == 0:  # Print 5% increments
                percent = (i / total * 100) if total > 0 else 0
                print(f"\r{desc}: {percent:.0f}% ({i}/{total})", end="", flush=True)
            yield item
        print(f"\r{desc}: 100% ({total}/{total})", flush=True)


def _normalize_text_for_hash(question: str, choices: list[str]) -> str:
    q = (question or "").strip().lower()
    ch = [c.strip().lower() for c in (choices or [])]
    # Collapse whitespace and remove excessive punctuation for hashing
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^a-z0-9_\s]", "", q)
    ch = [re.sub(r"\s+", " ", re.sub(r"[^a-z0-9_\s]", "", c)) for c in ch]
    return q + " || " + " | ".join(ch)


def _sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _simhash64(text: str) -> int:
    """Compute 64-bit SimHash for tokenized text.

    Uses SHA256(token) for better security and takes lower 64 bits.
    """
    if not text:
        return 0
    tokens = list(TOKEN_RE.findall(text))
    if not tokens:
        return 0
    # Term frequency weights
    tf: dict[str, int] = defaultdict(int)
    for t in tokens:
        tf[t] += 1
    acc = [0] * 64
    for t, w in tf.items():
        # Use SHA256 instead of MD5 for better security
        h = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[-8:], "big", signed=False)
        for i in range(64):
            bit = (h >> i) & 1
            acc[i] += w if bit == 1 else -w
    out = 0
    for i in range(64):
        if acc[i] > 0:
            out |= (1 << i)
    return out


def _hamming64(a: int, b: int) -> int:
    return int(bin((a ^ b) & ((1 << 64) - 1)).count("1"))


def _answer_to_index(answer: object, choices: list[str]) -> int | None:
    # Int index
    if isinstance(answer, int):
        return answer
    # String index (A/B/C...) or value match
    if isinstance(answer, str):
        s = answer.strip()
        # Letter index
        if len(s) == 1 and s.upper().isalpha():
            return ord(s.upper()) - ord("A")
        # Try exact or case-insensitive value match
        try:
            return choices.index(s)
        except ValueError:
            lower_map = {c.lower(): i for i, c in enumerate(choices)}
            return lower_map.get(s.lower())
    return None


def _has_duplicate_choices(choices: list[str]) -> bool:
    seen = set()
    for c in choices:
        key = (c or "").strip().lower()
        if key in seen:
            return True
        seen.add(key)
    return False


def _grammar_issues(text: str, timeout_seconds: float = DEFAULT_REGEX_TIMEOUT) -> set[str]:
    """Check for grammar/style issues with timeout protection."""
    issues: set[str] = set()
    if not text:
        return issues

    try:
        # Non-ASCII ratio
        non_ascii = sum(1 for ch in text if ord(ch) > 126)
        if non_ascii > 0 and non_ascii / max(1, len(text)) > 0.1:
            issues.add("NONASCII")

        # Use timeout for regex operations to prevent ReDoS
        with timeout_handler(timeout_seconds):
            # Repeated punctuation
            if re.search(r"[!?.,:;'-]{3,}", text):
                issues.add("REPEAT_PUNCT")
            # Multiple spaces
            if re.search(r"\s{2,}", text):
                issues.add("MULTISPACE")
            # Missing terminal punctuation for question-like prompts
            if len(text.strip()) > 12 and not re.search(r"[.?!]$", text.strip()):
                issues.add("NO_TERMINAL_PUNCT")

        # Unbalanced quotes
        if text.count("\"") % 2 != 0 or text.count("'") % 2 != 0:
            issues.add("UNBALANCED_QUOTES")

        # Unbalanced brackets (simple counter-based)
        pairs = [("(", ")"), ("[", "]"), ("{", "}")]
        for left, right in pairs:
            if text.count(left) != text.count(right):
                issues.add("UNBALANCED_BRACKETS")
                break

        # Leading lowercase (for question-like strings)
        first = next((ch for ch in text.strip() if ch.isalpha()), "")
        if first and first.islower():
            issues.add("LEADING_LOWER")

    except RegexTimeoutError:
        logger.warning("Regex timeout while checking grammar issues")
        issues.add("REGEX_TIMEOUT")
    except Exception as e:
        logger.warning(f"Error checking grammar issues: {e}")
        issues.add("GRAMMAR_CHECK_ERROR")

    return issues


@dataclass
class ItemHygiene:
    id: str
    simhash: int
    exact_hash: str
    dup_cluster: str | None
    dup_count: int
    bad_label: bool
    bad_label_reason: str | None
    choice_dup: bool
    issues_n: int
    issue_codes: str


def check_dataset_hygiene(
    records: Iterable[dict],
    dup_hamming: int = 3,
    show_progress: bool = True,
    validate_input: bool = True,
    max_size: int = MAX_DATASET_SIZE
) -> tuple[list[ItemHygiene], dict]:
    """Check dataset hygiene with validation and progress tracking.

    Args:
        records: Iterable of dataset records
        dup_hamming: Hamming distance threshold for near-duplicates
        show_progress: Whether to show progress bars
        validate_input: Whether to validate input records
        max_size: Maximum dataset size before warning

    Returns:
        Tuple of (hygiene results, summary dict)

    Raises:
        ValueError: If input validation fails
        Warning: If dataset size exceeds recommended limits
    """
    # Convert to list and check size
    records_list = list(records)
    n_records = len(records_list)

    if n_records == 0:
        logger.warning("Empty dataset provided")
        return [], {
            "n_items": 0,
            "dup_items": 0,
            "dup_clusters": 0,
            "bad_label_count": 0,
            "choice_dup_count": 0,
            "issues_items": 0,
            "dup_frac": 0.0,
            "bad_label_frac": 0.0,
            "choice_dup_frac": 0.0,
            "issues_frac": 0.0,
        }

    if n_records > max_size:
        logger.warning(
            f"Dataset size ({n_records}) exceeds recommended limit ({max_size}). "
            f"Performance may degrade. Consider chunking the dataset."
        )

    # First pass: validate and compute hashes and per-item checks
    items: list[dict] = []
    validation_errors = []
    # Process records with progress bar
    iterator = progress_bar(records_list, desc="Validating records") if show_progress else records_list

    for rec_idx, rec in enumerate(iterator):
        # Validate record if requested
        if validate_input:
            is_valid, error_msg = validate_record(rec)
            if not is_valid:
                validation_errors.append(f"Record {rec_idx}: {error_msg}")
                continue  # Skip invalid records

        try:
            iid = str(rec.get("id"))
            q = rec.get("question") or rec.get("input") or ""
            choices = list(rec.get("choices", []))
            answer = rec.get("answer")

            norm = _normalize_text_for_hash(str(q), [str(c) for c in choices])
            simh = _simhash64(norm)
            sha1 = _sha1_hex(norm)

            # Label checks
            idx = _answer_to_index(answer, [str(c) for c in choices])
            bad_label = False
            bad_reason: str | None = None
            if idx is None:
                bad_label = True
                bad_reason = "unparseable_answer"
            elif not (0 <= int(idx) < max(1, len(choices))):
                bad_label = True
                bad_reason = "answer_out_of_range"
            else:
                # If answer provided as string and not equal to index letter/value, that's ok.
                pass

            # Duplicate choices
            choice_dup = _has_duplicate_choices([str(c) for c in choices])

            # Grammar issues across question and choices
            issue_set = set()
            issue_set |= _grammar_issues(str(q))
            for c in choices:
                issue_set |= _grammar_issues(str(c))

            items.append(
                {
                    "id": iid,
                    "simhash": simh,
                    "exact_hash": sha1,
                    "bad_label": bad_label,
                    "bad_label_reason": bad_reason,
                    "choice_dup": choice_dup,
                    "issues": issue_set,
                }
            )
        except Exception as e:
            logger.error(f"Error processing record {rec_idx}: {e}")
            validation_errors.append(f"Record {rec_idx}: Processing error - {e}")

    # Report validation errors if any
    if validation_errors:
        logger.warning(f"Found {len(validation_errors)} invalid records")
        if len(validation_errors) <= 10:
            for error in validation_errors:
                logger.warning(error)
        else:
            logger.warning(f"First 10 errors: {validation_errors[:10]}")

    # Duplicate clustering: exact first, then simhash hamming threshold
    n = len(items)

    if n == 0:
        logger.warning("No valid records to process after validation")
        return [], {
            "n_items": 0,
            "dup_items": 0,
            "dup_clusters": 0,
            "bad_label_count": 0,
            "choice_dup_count": 0,
            "issues_items": 0,
            "dup_frac": 0.0,
            "bad_label_frac": 0.0,
            "choice_dup_frac": 0.0,
            "issues_frac": 0.0,
            "validation_errors": len(validation_errors),
        }

    logger.info(f"Processing {n} valid records for duplicate detection")
    # Build adjacency list
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    # Exact hash groups
    by_exact: dict[str, list[int]] = defaultdict(list)
    for i, it in enumerate(items):
        by_exact[it["exact_hash"].__str__()].append(i)
    for idxs in by_exact.values():
        if len(idxs) > 1:
            for i in idxs:
                for j in idxs:
                    if i != j:
                        adj[i].add(j)
    # Simhash near-duplicates with progress tracking
    sims = [it["simhash"] for it in items]

    # Show progress for O(n²) operation if dataset is large
    if show_progress and n > 100:
        total_comparisons = n * (n - 1) // 2
        comparison_count = 0
        print(f"\nComparing {total_comparisons:,} pairs for near-duplicates...")

    for i in range(n):
        for j in range(i + 1, n):
            if _hamming64(sims[i], sims[j]) <= dup_hamming:
                adj[i].add(j)
                adj[j].add(i)

            # Update progress for large datasets
            if show_progress and n > 100:
                comparison_count += 1
                if comparison_count % max(1, total_comparisons // 20) == 0:
                    percent = (comparison_count / total_comparisons * 100)
                    print(f"\rDuplicate detection: {percent:.0f}%", end="", flush=True)

    if show_progress and n > 100:
        print("\rDuplicate detection: 100%", flush=True)

    # Connected components
    visited = [False] * n
    clusters: list[list[int]] = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        comp: list[int] = []
        visited[i] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        clusters.append(comp)

    # Map to results
    index_to_cluster_id: dict[int, str] = {}
    for comp in clusters:
        if len(comp) == 1:
            idx = comp[0]
            index_to_cluster_id[idx] = ""
            continue
        # cluster label: smallest id lexicographically
        ids = [items[i]["id"] for i in comp]
        label = min(ids)
        for i in comp:
            index_to_cluster_id[i] = label

    results: list[ItemHygiene] = []
    for i, it in enumerate(items):
        cid = index_to_cluster_id.get(i, "")
        # cluster size
        dup_count = 0
        if cid:
            dup_count = sum(1 for j, jt in enumerate(items) if index_to_cluster_id.get(j, "") == cid)
        results.append(
            ItemHygiene(
                id=str(it["id"]),
                simhash=int(it["simhash"]),
                exact_hash=str(it["exact_hash"]),
                dup_cluster=cid or None,
                dup_count=dup_count,
                bad_label=bool(it["bad_label"]),
                bad_label_reason=it["bad_label_reason"],
                choice_dup=bool(it["choice_dup"]),
                issues_n=len(it["issues"]),
                issue_codes=",".join(sorted(it["issues"])) if it["issues"] else "",
            )
        )

    # Summary
    n_items = len(results)
    dup_items = sum(1 for r in results if r.dup_cluster)
    dup_clusters = len({r.dup_cluster for r in results if r.dup_cluster})
    bad_labels = sum(1 for r in results if r.bad_label)
    choice_dups = sum(1 for r in results if r.choice_dup)
    issues_items = sum(1 for r in results if r.issues_n > 0)
    summary = {
        "n_items": n_items,
        "dup_items": dup_items,
        "dup_clusters": dup_clusters,
        "bad_label_count": bad_labels,
        "choice_dup_count": choice_dups,
        "issues_items": issues_items,
        "dup_frac": (dup_items / n_items) if n_items else 0.0,
        "bad_label_frac": (bad_labels / n_items) if n_items else 0.0,
        "choice_dup_frac": (choice_dups / n_items) if n_items else 0.0,
        "issues_frac": (issues_items / n_items) if n_items else 0.0,
        "validation_errors": len(validation_errors),
        "performance_note": f"Processed {n_items} items" + (f" (O(n²) = {n_items**2:,} comparisons)" if n_items > 1000 else ""),
    }
    return results, summary


def write_report_csv(out_path: str | Path, results: list[ItemHygiene]) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "dup_cluster",
                "dup_count",
                "bad_label",
                "bad_label_reason",
                "choice_dup",
                "issues_n",
                "issue_codes",
                "simhash_hex",
                "exact_sha1",
            ]
        )
        for r in results:
            w.writerow(
                [
                    r.id,
                    r.dup_cluster or "",
                    r.dup_count,
                    int(r.bad_label),
                    r.bad_label_reason or "",
                    int(r.choice_dup),
                    r.issues_n,
                    r.issue_codes,
                    f"{r.simhash:016x}",
                    r.exact_hash,
                ]
            )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Automated QA & label hygiene checks",
        epilog="Performance: Optimal for <10K items. O(n²) complexity for duplicate detection."
    )
    ap.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    ap.add_argument(
        "--out-csv",
        default="artifacts/data_quality_report.csv",
        help="Path for public-safe CSV report",
    )
    ap.add_argument("--dup-hamming", type=int, default=3, help="Hamming threshold for near-duplicates")
    ap.add_argument("--max-dup-frac", type=float, default=0.05, help="Fail if fraction of items in duplicate clusters exceeds this")
    ap.add_argument("--max-bad-label-frac", type=float, default=0.0, help="Fail if fraction of bad labels exceeds this")
    ap.add_argument("--max-choice-dup-frac", type=float, default=0.02, help="Fail if fraction of items with duplicate choices exceeds this")
    ap.add_argument("--max-issues-frac", type=float, default=0.3, help="Warn/fail if fraction with grammar issues exceeds this")
    ap.add_argument("--strict", action="store_true", help="Treat grammar issues threshold as failure (not warning)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    ap.add_argument("--no-validation", action="store_true", help="Skip input validation")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    ap.add_argument("--max-size", type=int, default=MAX_DATASET_SIZE, help="Warn if dataset exceeds this size")

    args = ap.parse_args(argv)

    # Configure logging based on verbosity
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logger.setLevel(logging.INFO)

    dataset_path = Path(args.dataset)

    # Check if file exists
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        return 1

    # Check file size
    if dataset_path.stat().st_size > MAX_INPUT_SIZE:
        logger.error(f"Dataset file too large (>{MAX_INPUT_SIZE/1024/1024:.1f}MB)")
        return 1

    start_time = time.time()
    try:
        rows = list(read_jsonl(dataset_path))
    except Exception as e:
        logger.error(f"Failed to read dataset: {e}")
        return 1

    logger.info(f"Loaded {len(rows)} records from {dataset_path}")

    results, summary = check_dataset_hygiene(
        rows,
        dup_hamming=int(args.dup_hamming),
        show_progress=not args.no_progress,
        validate_input=not args.no_validation,
        max_size=args.max_size
    )
    write_report_csv(args.out_csv, results)

    elapsed_time = time.time() - start_time

    # Add timing to summary
    summary["elapsed_seconds"] = round(elapsed_time, 2)
    summary["items_per_second"] = round(len(rows) / elapsed_time, 1) if elapsed_time > 0 else 0

    # Print machine-readable summary
    print(json.dumps({"summary": summary, "report": str(Path(args.out_csv))}))

    if args.verbose:
        logger.info(f"Processing completed in {elapsed_time:.2f} seconds")
        logger.info(f"Report written to {args.out_csv}")

    # Threshold enforcement
    fail = False
    if summary["dup_frac"] > float(args.max_dup_frac):
        fail = True
    if summary["bad_label_frac"] > float(args.max_bad_label_frac):
        fail = True
    if summary["choice_dup_frac"] > float(args.max_choice_dup_frac):
        fail = True
    if args.strict and (summary["issues_frac"] > float(args.max_issues_frac)):
        fail = True
    return 1 if fail else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

