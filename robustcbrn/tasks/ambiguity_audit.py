"""Ambiguity/unanswerable audit task with enhanced features."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .common import load_mcq_dataset
from ..budget_guard import BudgetGuard
from ..config import get_paths
from ..qa.ambiguity import (
    audit_dataset,
    decisions_to_records,
    AmbiguityConfig,
    AmbiguityDetectionError,
    DEFAULT_CONFIG
)

# Configure logging
logger = logging.getLogger(__name__)


def run_ambiguity_audit(
    dataset_path: str,
    mode: str = "heuristic",
    seed: int = 123,
    max_items: Optional[int] = None,
    out_path: Optional[str] = None,
    config: Optional[AmbiguityConfig] = None,
    collect_metrics: bool = True,
    verbose: bool = False,
) -> str:
    """Run ambiguity/unanswerable audit and write Inspect-like JSON log.

    Modes:
      - heuristic: safe, metadata-only heuristics over choices
      - llm: placeholder; not enabled by default in OSS

    Args:
        dataset_path: Path to MCQ JSONL dataset
        mode: Audit mode ('heuristic' or 'llm')
        seed: Random seed for reproducibility
        max_items: Maximum number of items to process
        out_path: Explicit output path for JSON log
        config: Configuration for ambiguity detection
        collect_metrics: Whether to collect performance metrics
        verbose: Enable verbose logging

    Returns:
        Path to the output JSON file

    Raises:
        AmbiguityDetectionError: If processing fails
        RuntimeError: If LLM mode is selected (disabled)
        ValueError: If invalid mode is specified
    """
    # Configure logging level
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    logger.info(f"Starting ambiguity audit on {dataset_path}")

    # Load configuration
    if config is None:
        config = DEFAULT_CONFIG
        logger.debug("Using default configuration")

    # Load dataset
    try:
        ds = load_mcq_dataset(dataset_path, shuffle_seed=None, max_items=max_items)
        samples = list(ds)  # type: ignore[arg-type]
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        samples = ds  # type: ignore[assignment]

    logger.info(f"Loaded {len(samples)} samples from dataset")

    model_tag = "heuristic" if mode != "llm" else "llm-critic"

    # Safe, no-op budget guard (dry-run) to keep interface consistent
    with BudgetGuard(job_name="ambiguity_audit", projected_hours=0.0, projected_api_usd=0.0, dry_run=True):
        if mode == "heuristic":
            logger.info("Running heuristic-based audit")
            try:
                decisions, metrics = audit_dataset(
                    samples,
                    config=config,
                    collect_metrics=collect_metrics
                )
            except AmbiguityDetectionError as e:
                logger.error(f"Audit failed: {e}")
                raise
        elif mode == "llm":
            # Disabled by default; keep interface for future sanitized critic integration
            raise RuntimeError(
                "LLM critic mode is disabled by default. Provide a local sanitized critic to enable."
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    # Convert decisions to records
    rows = decisions_to_records(decisions)

    # Build output payload
    payload: Dict[str, Any] = {
        "task": "ambiguity_audit",
        "model": model_tag,
        "seed": seed,
        "samples": rows,
    }

    # Add metrics if collected
    if collect_metrics and metrics:
        payload["metrics"] = metrics.to_dict()
        logger.info(f"Audit metrics: {metrics.to_dict()}")

    # Add configuration info
    payload["config"] = {
        "jaccard_threshold": config.jaccard_threshold,
        "numeric_proximity_threshold": config.numeric_proximity_threshold,
        "max_tokens_for_boolean": config.max_tokens_for_boolean,
    }

    # Write output
    out_dir = Path(get_paths().logs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_path is None:
        base = Path(dataset_path).stem
        out_path = str(out_dir / f"ambiguity_audit_{base}.json")

    Path(out_path).write_text(json.dumps(payload, indent=2))
    logger.info(f"Audit results written to {out_path}")

    # Log summary statistics
    if collect_metrics and metrics:
        logger.info(f"Summary: {metrics.clean_count} clean, "
                   f"{metrics.ambiguous_count} ambiguous, "
                   f"{metrics.unanswerable_count} unanswerable")

    return out_path


def cli(argv: Optional[list[str]] = None) -> int:
    """Command-line interface for ambiguity audit."""
    ap = argparse.ArgumentParser(
        description="Ambiguity/unanswerable audit with configurable thresholds",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("dataset_path", help="Path to MCQ JSONL dataset")
    ap.add_argument("--mode", choices=["heuristic", "llm"], default="heuristic",
                   help="Audit mode")
    ap.add_argument("--seed", type=int, default=123,
                   help="Random seed for reproducibility")
    ap.add_argument("--max-items", type=int, default=None,
                   help="Maximum number of items to process")
    ap.add_argument("--out", type=str, default=None,
                   help="Explicit output JSON path")

    # Configuration options
    ap.add_argument("--jaccard-threshold", type=float, default=0.9,
                   help="Jaccard similarity threshold for near-duplicates")
    ap.add_argument("--numeric-threshold", type=float, default=0.01,
                   help="Numeric proximity threshold (as fraction)")
    ap.add_argument("--max-boolean-tokens", type=int, default=2,
                   help="Max tokens to consider as boolean-like")

    # Feature flags
    ap.add_argument("--no-metrics", action="store_true",
                   help="Disable performance metrics collection")
    ap.add_argument("--verbose", "-v", action="store_true",
                   help="Enable verbose logging")

    args = ap.parse_args(argv)

    # Build configuration from CLI args
    config = AmbiguityConfig(
        jaccard_threshold=args.jaccard_threshold,
        numeric_proximity_threshold=args.numeric_threshold,
        max_tokens_for_boolean=args.max_boolean_tokens
    )

    try:
        out = run_ambiguity_audit(
            dataset_path=args.dataset_path,
            mode=args.mode,
            seed=args.seed,
            max_items=args.max_items,
            out_path=args.out,
            config=config,
            collect_metrics=not args.no_metrics,
            verbose=args.verbose,
        )
        print(f"Audit complete: {out}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(cli())