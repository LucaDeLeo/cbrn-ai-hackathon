from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from robustcbrn.analysis.heuristics import analyze_questions
from robustcbrn.config import AppConfig
from robustcbrn.data.loader import load_dataset
from robustcbrn.statistical.position_bias import detect_position_bias
from robustcbrn.utils.determinism import set_determinism
from robustcbrn.utils.logging import setup_logging


def validate_analyze_inputs(input_path: str, output_path: str | None, logger) -> tuple[Path, Path | None, int]:
    """Validate input and output paths for analyze command.

    Args:
        input_path: Path to input dataset file
        output_path: Optional path to output file
        logger: Logger instance for error logging

    Returns:
        Tuple of (validated_input_path, validated_output_path, exit_code)
        exit_code is 0 for success, 1 for error
    """
    # Validate input file exists
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        print(f"Error: Input file '{input_path_obj}' not found")
        logger.error("FileNotFoundError: Input file '%s' not found", input_path_obj)
        return input_path_obj, None, 1

    if not input_path_obj.is_file():
        print(f"Error: Input path '{input_path_obj}' is not a file")
        logger.error("ValueError: Input path '%s' is not a file", input_path_obj)
        return input_path_obj, None, 1

    # Validate output directory if specified
    output_path_obj = None
    if output_path:
        output_path_obj = Path(output_path)
        output_dir = output_path_obj.parent
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = output_dir / ".write_test"
            test_file.touch()
            test_file.unlink()
        except PermissionError:
            print(f"Error: Permission denied accessing '{output_dir}'")
            logger.error("PermissionError: Cannot write to output directory '%s'", output_dir)
            return input_path_obj, None, 1

    return input_path_obj, output_path_obj, 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RobustCBRN CLI - Toolkit to robustify CBRN MCQA benchmarks",
        epilog="""Examples:
  # Load and preview a dataset
  python -m robustcbrn.cli.main load data/wmdp_bio_sample_100.jsonl

  # Analyze dataset with default settings
  python -m robustcbrn.cli.main analyze data/wmdp_bio_sample_100.jsonl --output results/analysis.json

  # Dry run to validate inputs without processing
  python -m robustcbrn.cli.main analyze data/wmdp_bio_sample_100.jsonl --dry-run

  # Analyze with verbose output and limit to 50 questions
  python -m robustcbrn.cli.main analyze data/dataset.jsonl --verbose --max-items 50

  # Analyze with custom configuration
  python -m robustcbrn.cli.main analyze data/dataset.jsonl --config configs/custom.json --output results/report.json

  # Analyze position bias in dataset
  python -m robustcbrn.cli.main position-bias data/dataset.jsonl --output results/position_bias.json --verbose
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Load command (existing)
    load_parser = subparsers.add_parser("load", help="Load and preview a dataset")
    load_parser.add_argument("path", help="Path to dataset file")
    load_parser.add_argument("--config", default="configs/default.json", help="Path to config JSON (default: configs/default.json)")
    load_parser.add_argument("--id-salt", default=None, help="Override ID salt for hashing (optional)")

    # Analyze command (new)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a dataset using heuristics")
    analyze_parser.add_argument("--input", "-i", required=True, help="Input dataset file path")
    analyze_parser.add_argument("--output", "-o", help="Output file path for results (JSON format)")
    analyze_parser.add_argument("--config", "-c", default="configs/default.json", help="Configuration file path (default: configs/default.json)")
    analyze_parser.add_argument("--dry-run", action="store_true", help="Validate inputs without processing")
    analyze_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed progress during analysis")
    analyze_parser.add_argument("--models", nargs="+", help="Models to use for analysis (not yet implemented)")
    analyze_parser.add_argument("--max-items", type=int, help="Maximum number of questions to process")
    analyze_parser.add_argument("--time-limit", type=int, help="Time limit in seconds (not yet implemented)")
    analyze_parser.add_argument("--budget", type=float, help="Budget limit for analysis (not yet implemented)")
    analyze_parser.add_argument("--public-report", choices=["on", "off"], default="off", help="Generate public report (always generates full report currently)")
    analyze_parser.add_argument("--id-salt", default=None, help="Override ID salt for hashing (optional)")
    analyze_parser.add_argument("--robust-input", help="Path to robust dataset for degradation analysis")
    analyze_parser.add_argument("--stratify-by", help="Path to stratification metadata (JSON file with question_id -> stratum mapping)")
    analyze_parser.add_argument("--tests", nargs="+", help="Specific tests to run (e.g., position_bias, lexical_patterns, heuristic_degradation)")

    # Position Bias Analysis command (Epic 2, Story 2.1)
    position_parser = subparsers.add_parser("position-bias", help="Analyze position bias in MCQA dataset")
    position_parser.add_argument("--input", "-i", required=True, help="Input dataset file path")
    position_parser.add_argument("--output", "-o", help="Output file path for results (JSON format)")
    position_parser.add_argument("--config", "-c", default="configs/default.json", help="Configuration file path")
    position_parser.add_argument("--significance", type=float, default=0.05, help="Significance level for statistical tests (default: 0.05)")
    position_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    position_parser.add_argument("--id-salt", default=None, help="Override ID salt for hashing")

    # Heuristic Degradation Analysis command
    degradation_parser = subparsers.add_parser("heuristic-degradation", help="Analyze heuristic degradation between original and robust datasets")
    degradation_parser.add_argument("--original", "-i", required=True, help="Path to original dataset")
    degradation_parser.add_argument("--robust", "-r", required=True, help="Path to robust dataset")
    degradation_parser.add_argument("--output", "-o", help="Output file path for results")
    degradation_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    degradation_parser.add_argument("--config", "-c", default="configs/default.json", help="Configuration file path")
    degradation_parser.add_argument("--id-salt", default=None, help="Override ID salt for hashing")

    args = parser.parse_args()

    # Handle case where no command is provided
    if args.command is None:
        parser.print_help()
        return 1

    # Get config path based on command
    config_path = args.config if hasattr(args, "config") else "configs/default.json"

    try:
        cfg = AppConfig.from_json(config_path)
    except FileNotFoundError:
        print(f"Error: Config file '{config_path}' not found")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{config_path}': {e}")
        return 1
    logger = setup_logging(cfg.logging.log_dir, cfg.logging.filename, cfg.logging.level)
    set_determinism(
        seed=cfg.determinism.seed,
        cudnn_deterministic=cfg.determinism.cudnn_deterministic,
        cudnn_benchmark=cfg.determinism.cudnn_benchmark,
        cublas_workspace=cfg.determinism.cublas_workspace,
        python_hash_seed=cfg.determinism.python_hash_seed,
        tokenizers_parallelism=cfg.determinism.tokenizers_parallelism,
    )

    try:
        if args.command == "load":
            if not args.path:
                parser.error("load requires a dataset path")
                return 1
            ds = load_dataset(
                args.path,
                csv_mapping=cfg.data.csv_mapping,
                id_salt=(args.id_salt if args.id_salt is not None else cfg.data.id_salt),
            )
            logger.info("Loaded %d records from %s", len(ds), args.path)
            # Print a short preview
            for it in ds[:3]:
                logger.info("%s | %s | choices=%d | answer=%d", it.id, it.question[:50].replace("\n", " "), len(it.choices), it.answer)
            return 0

        elif args.command == "analyze":
            # Validate input and output paths
            input_path, output_path, exit_code = validate_analyze_inputs(
                args.input, args.output, logger
            )
            if exit_code != 0:
                return exit_code

            # Load and validate dataset
            try:
                questions = load_dataset(
                    str(input_path),
                    csv_mapping=cfg.data.csv_mapping,
                    id_salt=(args.id_salt if args.id_salt is not None else cfg.data.id_salt),
                )
            except FileNotFoundError:
                print(f"Error: Input file '{input_path}' not found")
                logger.error("FileNotFoundError during load: Input file '%s' not found", input_path)
                return 1
            except PermissionError:
                print(f"Error: Permission denied accessing '{input_path}'")
                logger.error("PermissionError during load: Cannot read input file '%s'", input_path)
                return 1
            except json.JSONDecodeError as e:
                print(f"Error: Invalid JSON format in '{input_path}'")
                logger.error("JSONDecodeError during load: Invalid JSON in '%s': %s", input_path, str(e), exc_info=True)
                return 1
            except ValueError as e:
                print(f"Error: Invalid data format - {e}")
                logger.error("ValueError during load: Invalid data format in '%s': %s", input_path, str(e), exc_info=True)
                return 1

            # Apply --max-items limit if specified
            original_count = len(questions)
            if args.max_items and args.max_items > 0:
                questions = questions[:args.max_items]
                if args.verbose:
                    print(f"Limited dataset from {original_count} to {len(questions)} questions")

            # Dry run - validate and exit
            if args.dry_run:
                print("Validation successful:")
                print(f"  - Input file: {input_path}")
                print("  - Dataset format: Valid")
                print(f"  - Questions loaded: {len(questions)}")
                if output_path:
                    print(f"  - Output directory: {output_path.parent} (writable)")
                else:
                    print("  - Output: None specified (results will not be saved)")
                logger.info("Dry run validation successful for '%s'", input_path)
                return 0

            # Handle not-yet-implemented flags with informative messages
            if args.models:
                print("Model selection not yet implemented")
                logger.info("User requested models: %s (not yet implemented)", args.models)

            if args.time_limit:
                print("Time limit not yet implemented")
                logger.info("User requested time limit: %d seconds (not yet implemented)", args.time_limit)

            if args.budget:
                print("Budget limits not yet implemented")
                logger.info("User requested budget: %.2f (not yet implemented)", args.budget)

            if args.public_report == "on":
                logger.info("Public report requested (currently always generates full report)")

            # Load robust dataset if provided
            robust_questions = None
            if args.robust_input:
                try:
                    robust_questions = load_dataset(
                        args.robust_input,
                        csv_mapping=cfg.data.csv_mapping,
                        id_salt=(args.id_salt if args.id_salt is not None else cfg.data.id_salt),
                    )
                    if args.verbose:
                        print(f"Loaded {len(robust_questions)} robust questions from {args.robust_input}")
                except Exception as e:
                    print(f"Error loading robust dataset: {e}")
                    logger.error("Error loading robust dataset: %s", str(e), exc_info=True)
                    robust_questions = None

            # Load stratification metadata if provided
            stratify_by = None
            if args.stratify_by:
                try:
                    import numpy as np
                    with open(args.stratify_by) as f:
                        stratification_data = json.load(f)

                    # Create stratify_by array aligned with questions
                    stratify_by = np.array([stratification_data.get(q.id, "unknown") for q in questions])

                    if args.verbose:
                        unique_strata = np.unique(stratify_by)
                        print(f"Loaded stratification data with {len(unique_strata)} strata: {list(unique_strata)}")
                except Exception as e:
                    print(f"Error loading stratification data: {e}")
                    logger.error("Error loading stratification data: %s", str(e), exc_info=True)
                    stratify_by = None

            # Run analysis
            if args.verbose:
                print(f"Analyzing {len(questions)} questions from {input_path}...")

            report = analyze_questions(
                questions=questions,
                show_progress=args.verbose,
                save_path=output_path,
                dataset_path=input_path,
                dataset_hash=None,  # Could compute hash if needed
                debug=False,
                tests_to_run=args.tests,
                robust_questions=robust_questions,
                stratify_by=stratify_by  # Add this parameter
            )

            # Print summary
            print("\nAnalysis complete:")
            print(f"  Method: {report.method}")
            print(f"  Accuracy: {report.results['accuracy']:.2%}")
            print(f"  Correct: {report.results['correct_predictions']}/{report.results['total_predictions']}")
            print(f"  Runtime: {report.performance['runtime_seconds']:.2f}s")
            if output_path:
                print(f"  Results saved to: {output_path}")

            # Display degradation analysis results if robust questions were provided
            if robust_questions and 'heuristic_degradation' in report.results:
                degradation_data = report.results['heuristic_degradation']
                print("\n📊 Heuristic Degradation Analysis:")
                print("═══════════════════════════════════")
                summary = degradation_data.get('summary', {})
                print(f"  - Total heuristics: {summary.get('total_heuristics', 0)}")
                print(f"  - Significant degradations: {summary.get('significant_degradations', 0)}")
                print(f"  - Average degradation: {summary.get('average_degradation', 0):.2%}")
                print(f"  - Maximum degradation: {summary.get('maximum_degradation', 0):.2%}")
                print(f"  - Degradation percentage: {summary.get('degradation_percentage', 0):.1f}%")

                if args.verbose:
                    print("\nDetailed Heuristic Results:")
                    heuristics = degradation_data.get('heuristics', {})
                    for heuristic_name, heuristic_result in heuristics.items():
                        print(f"  {heuristic_name.replace('_', ' ').title()}:")
                        print(f"    Original accuracy: {heuristic_result.get('original_accuracy', 0):.2%}")
                        print(f"    Robust accuracy: {heuristic_result.get('robust_accuracy', 0):.2%}")
                        print(f"    Absolute delta: {heuristic_result.get('absolute_delta', 0):.2%}")
                        print(f"    Significant: {'Yes' if heuristic_result.get('is_significant', False) else 'No'}")

            logger.info(
                "Analysis completed: accuracy=%.2f%%, correct=%d/%d, runtime=%.2fs",
                report.results['accuracy'] * 100,
                report.results['correct_predictions'],
                report.results['total_predictions'],
                report.performance['runtime_seconds']
            )

            return 0

        elif args.command == "position-bias":
            # Validate input file
            input_path = Path(args.input)
            if not input_path.exists():
                print(f"Error: Input file '{input_path}' not found")
                logger.error("FileNotFoundError: Input file '%s' not found", input_path)
                return 1

            # Load dataset
            try:
                questions = load_dataset(
                    str(input_path),
                    csv_mapping=cfg.data.csv_mapping,
                    id_salt=(args.id_salt if args.id_salt is not None else cfg.data.id_salt),
                )
            except Exception as e:
                print(f"Error loading dataset: {e}")
                logger.error("Error loading dataset: %s", str(e), exc_info=True)
                return 1

            if args.verbose:
                print(f"Loaded {len(questions)} questions from {input_path}")

            # Run position bias analysis
            try:
                # Convert questions to dictionaries for position bias analysis
                questions_dict = []
                for q in questions:
                    q_dict = {
                        'id': q.id,
                        'question': q.question,
                        'choices': q.choices,
                        'answer_index': q.answer,
                    }
                    questions_dict.append(q_dict)

                results = detect_position_bias(questions_dict)

                # Compute total variants from the new report structure
                total_variants_generated = sum(
                    len(v) for v in results.get("position_swaps", {}).values()
                )

                # Display results
                print("\n🔍 Position Bias Analysis Results")
                print("═══════════════════════════════════")
                print(f"Dataset: {len(questions)} questions")
                print(f"Observed frequencies: {results['observed_frequencies']}")
                print(f"Expected frequencies: {results['expected_frequencies']}")
                print(f"Chi-square statistic: {results['chi_square_statistic']:.4f}")
                print(f"P-value: {results['p_value']:.6f}")
                print(f"Effect size (Cramer's V): {results['effect_size']:.4f}")
                print(f"Significant bias detected: {'YES' if results['significant'] else 'NO'}")
                print(f"Predictive questions found: {len(results['predictive_questions'])}")
                print(f"Position swap variants generated: {total_variants_generated}")

                if args.verbose and results['predictive_questions']:
                    print("\nPredictive Question IDs (first 10):")
                    for qid in results['predictive_questions'][:10]:
                        print(f"  - {qid}")

                if args.output:
                    print(f"\n💾 Detailed results saved to: {args.output}")

                logger.info(
                    "Position bias analysis completed: bias_detected=%s, predictive_questions=%d",
                    results['significant'],
                    len(results.get('predictive_questions', []))
                )

                return 0

            except Exception as e:
                print(f"Error during position bias analysis: {e}")
                logger.error("Error during position bias analysis: %s", str(e), exc_info=True)
                return 1

        elif args.command == "heuristic-degradation":
            # Load original dataset
            try:
                original_questions = load_dataset(
                    args.original,
                    csv_mapping=cfg.data.csv_mapping,
                    id_salt=(args.id_salt if args.id_salt is not None else cfg.data.id_salt),
                )
            except Exception as e:
                print(f"Error loading original dataset: {e}")
                logger.error("Error loading original dataset: %s", str(e), exc_info=True)
                return 1

            # Load robust dataset
            try:
                robust_questions = load_dataset(
                    args.robust,
                    csv_mapping=cfg.data.csv_mapping,
                    id_salt=(args.id_salt if args.id_salt is not None else cfg.data.id_salt),
                )
            except Exception as e:
                print(f"Error loading robust dataset: {e}")
                logger.error("Error loading robust dataset: %s", str(e), exc_info=True)
                return 1

            if args.verbose:
                print(f"Analyzing heuristic degradation between {len(original_questions)} original and {len(robust_questions)} robust questions...")

            # Run the analysis
            from robustcbrn.analysis.heuristic_degradation import HeuristicDegradationAnalyzer
            analyzer = HeuristicDegradationAnalyzer()
            result = analyzer.analyze(
                [q.__dict__ if hasattr(q, '__dict__') else q for q in original_questions],
                [q.__dict__ if hasattr(q, '__dict__') else q for q in robust_questions]
            )

            # Save results if requested
            if args.output:
                analyzer.save_json(result, args.output)
                print(f"Results saved to: {args.output}")

            # Display summary
            print("\n📊 Heuristic Degradation Analysis Results")
            print("══════════════════════════════════════════")
            print(f"Original questions: {result.total_original}")
            print(f"Robust questions: {result.total_robust}")
            print(f"Runtime: {result.runtime_seconds:.2f}s")
            print("\nSummary:")
            print(f"  - Total heuristics: {result.summary['total_heuristics']}")
            print(f"  - Significant degradations: {result.summary['significant_degradations']}")
            print(f"  - Average degradation: {result.summary['average_degradation']:.2%}")
            print(f"  - Maximum degradation: {result.summary['maximum_degradation']:.2%}")
            print(f"  - Degradation percentage: {result.summary['degradation_percentage']:.1f}%")

            if args.verbose:
                print("\nDetailed Results:")
                for heuristic_name, heuristic_result in result.heuristics.items():
                    print(f"\n  {heuristic_name.replace('_', ' ').title()}:")
                    print(f"    Original accuracy: {heuristic_result.original_accuracy:.2%}")
                    print(f"    Robust accuracy: {heuristic_result.robust_accuracy:.2%}")
                    print(f"    Absolute delta: {heuristic_result.absolute_delta:.2%}")
                    print(f"    95% CI: [{heuristic_result.confidence_interval[0]:.2%}, {heuristic_result.confidence_interval[1]:.2%}]")
                    print(f"    Significant: {'Yes' if heuristic_result.is_significant else 'No'}")

            return 0

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
        logger.info("Analysis interrupted by user (KeyboardInterrupt)")
        return 1
    except Exception as e:
        print(f"Error: Unexpected error occurred - {e}")
        logger.error("Unexpected error: %s", str(e), exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
