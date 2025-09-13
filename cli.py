from __future__ import annotations

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple

from src.config import AppConfig
from src.utils.logging import setup_logging
from src.utils.determinism import set_determinism
from src.data.loader import load_dataset
from src.analysis.heuristics import analyze_questions
from src.statistical.position_bias import run_position_bias_analysis


def validate_analyze_inputs(input_path: str, output_path: Optional[str], logger) -> Tuple[
    Path, Optional[Path], int]:
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
        description="CBRN CLI - Toolkit to robustify CBRN MCQA benchmarks",
        epilog="""Examples:
  # Load and preview a dataset
  python cli.py load data/wmdp_bio_sample_100.jsonl

  # Analyze dataset with default settings
  python cli.py analyze data/wmdp_bio_sample_100.jsonl --output results/analysis.json

  # Dry run to validate inputs without processing
  python cli.py analyze data/wmdp_bio_sample_100.jsonl --dry-run

  # Analyze with verbose output and limit to 50 questions
  python cli.py analyze data/dataset.jsonl --verbose --max-items 50

  # Analyze with custom configuration
  python cli.py analyze data/dataset.jsonl --config configs/custom.json --output results/report.json

  # Analyze position bias in dataset
  python cli.py position-bias data/dataset.jsonl --output results/position_bias.json --verbose

  # Position bias with bootstrap confidence intervals
  python cli.py position-bias data/dataset.jsonl --bootstrap --n-bootstrap 10000 --confidence-level 0.95 --verbose

  # Test bootstrap implementation
  python cli.py bootstrap-test --data-type normal --n-samples 1000 --n-bootstrap 5000
""",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Load command (existing)
    load_parser = subparsers.add_parser("load", help="Load and preview a dataset")
    load_parser.add_argument("path", help="Path to dataset file")
    load_parser.add_argument("--config", default="configs/default.json",
                             help="Path to config JSON (default: configs/default.json)")
    load_parser.add_argument("--id-salt", default=None,
                             help="Override ID salt for hashing (optional)")

    # Analyze command (existing)
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a dataset using heuristics")
    analyze_parser.add_argument("--input", "-i", required=True, help="Input dataset file path")
    analyze_parser.add_argument("--output", "-o", help="Output file path for results (JSON format)")
    analyze_parser.add_argument("--config", "-c", default="configs/default.json",
                                help="Configuration file path (default: configs/default.json)")
    analyze_parser.add_argument("--dry-run", action="store_true",
                                help="Validate inputs without processing")
    analyze_parser.add_argument("--verbose", "-v", action="store_true",
                                help="Show detailed progress during analysis")
    analyze_parser.add_argument("--models", nargs="+",
                                help="Models to use for analysis (not yet implemented)")
    analyze_parser.add_argument("--max-items", type=int,
                                help="Maximum number of questions to process")
    analyze_parser.add_argument("--time-limit", type=int,
                                help="Time limit in seconds (not yet implemented)")
    analyze_parser.add_argument("--budget", type=float,
                                help="Budget limit for analysis (not yet implemented)")
    analyze_parser.add_argument("--public-report", choices=["on", "off"], default="off",
                                help="Generate public report (always generates full report currently)")
    analyze_parser.add_argument("--id-salt", default=None,
                                help="Override ID salt for hashing (optional)")

    # Position Bias Analysis command (Epic 2, Story 2.1 + 2.2)
    position_parser = subparsers.add_parser("position-bias",
                                            help="Analyze position bias in MCQA dataset with optional bootstrap CIs")
    position_parser.add_argument("--input", "-i", required=True, help="Input dataset file path")
    position_parser.add_argument("--output", "-o",
                                 help="Output file path for results (JSON format)")
    position_parser.add_argument("--config", "-c", default="configs/default.json",
                                 help="Configuration file path")
    position_parser.add_argument("--significance", type=float, default=0.05,
                                 help="Significance level for statistical tests (default: 0.05)")
    position_parser.add_argument("--verbose", "-v", action="store_true",
                                 help="Show detailed output")
    position_parser.add_argument("--id-salt", default=None, help="Override ID salt for hashing")

    # New bootstrap arguments for Story 2.2
    position_parser.add_argument("--bootstrap", action="store_true",
                                 help="Enable bootstrap confidence intervals")
    position_parser.add_argument("--n-bootstrap", type=int, default=10000,
                                 help="Number of bootstrap iterations (default: 10,000)")
    position_parser.add_argument("--confidence-level", type=float, default=0.95,
                                 help="Confidence level for bootstrap CIs (default: 0.95)")
    position_parser.add_argument("--bootstrap-method", choices=["percentile", "bca"],
                                 default="percentile",
                                 help="Bootstrap CI method: percentile or BCa (default: percentile)")
    position_parser.add_argument("--adaptive-bootstrap", action="store_true",
                                 help="Enable adaptive early stopping for bootstrap")

    # Bootstrap test command for testing/validation
    bootstrap_parser = subparsers.add_parser("bootstrap-test",
                                             help="Test bootstrap CI implementation")
    bootstrap_parser.add_argument("--data-type", choices=["normal", "uniform", "binary"],
                                  default="normal",
                                  help="Type of test data to generate")
    bootstrap_parser.add_argument("--n-samples", type=int, default=1000,
                                  help="Number of data samples")
    bootstrap_parser.add_argument("--n-bootstrap", type=int, default=5000,
                                  help="Bootstrap iterations")
    bootstrap_parser.add_argument("--confidence-level", type=float, default=0.95,
                                  help="Confidence level")
    bootstrap_parser.add_argument("--method", choices=["percentile", "bca"], default="percentile",
                                  help="Bootstrap method")
    bootstrap_parser.add_argument("--statistic", choices=["mean", "median", "std"], default="mean",
                                  help="Statistic to bootstrap")
    bootstrap_parser.add_argument("--verbose", "-v", action="store_true",
                                  help="Show detailed output")

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
                logger.info("%s | %s | choices=%d | answer=%d", it.id,
                            it.question[:50].replace("\n", " "), len(it.choices), it.answer)
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
                logger.error("JSONDecodeError during load: Invalid JSON in '%s': %s", input_path,
                             str(e), exc_info=True)
                return 1
            except ValueError as e:
                print(f"Error: Invalid data format - {e}")
                logger.error("ValueError during load: Invalid data format in '%s': %s", input_path,
                             str(e), exc_info=True)
                return 1

            # Apply --max-items limit if specified
            original_count = len(questions)
            if args.max_items and args.max_items > 0:
                questions = questions[:args.max_items]
                if args.verbose:
                    print(f"Limited dataset from {original_count} to {len(questions)} questions")

            # Dry run - validate and exit
            if args.dry_run:
                print(f"Validation successful:")
                print(f"  - Input file: {input_path}")
                print(f"  - Dataset format: Valid")
                print(f"  - Questions loaded: {len(questions)}")
                if output_path:
                    print(f"  - Output directory: {output_path.parent} (writable)")
                else:
                    print(f"  - Output: None specified (results will not be saved)")
                logger.info("Dry run validation successful for '%s'", input_path)
                return 0

            # Handle not-yet-implemented flags with informative messages
            if args.models:
                print("Model selection not yet implemented")
                logger.info("User requested models: %s (not yet implemented)", args.models)

            if args.time_limit:
                print("Time limit not yet implemented")
                logger.info("User requested time limit: %d seconds (not yet implemented)",
                            args.time_limit)

            if args.budget:
                print("Budget limits not yet implemented")
                logger.info("User requested budget: %.2f (not yet implemented)", args.budget)

            if args.public_report == "on":
                logger.info("Public report requested (currently always generates full report)")

            # Run analysis
            if args.verbose:
                print(f"Analyzing {len(questions)} questions from {input_path}...")

            report = analyze_questions(
                questions=questions,
                show_progress=args.verbose,
                save_path=output_path,
                dataset_path=input_path,
                dataset_hash=None,  # Could compute hash if needed
                debug=False
            )

            # Print summary
            print(f"\nAnalysis complete:")
            print(f"  Method: {report.method}")
            print(f"  Accuracy: {report.results['accuracy']:.2%}")
            print(
                f"  Correct: {report.results['correct_predictions']}/{report.results['total_predictions']}")
            print(f"  Runtime: {report.performance['runtime_seconds']:.2f}s")
            if output_path:
                print(f"  Results saved to: {output_path}")

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

            # Prepare analysis parameters
            analysis_kwargs = {
                "significance_level": args.significance,
                "save_path": Path(args.output) if args.output else None
            }

            # Add bootstrap parameters if enabled
            if args.bootstrap:
                analysis_kwargs.update({
                    "n_bootstrap": args.n_bootstrap,
                    "confidence_level": args.confidence_level,
                    "bootstrap_method": args.bootstrap_method
                })

                if args.verbose:
                    print(f"Bootstrap analysis enabled:")
                    print(f"  - Iterations: {args.n_bootstrap}")
                    print(f"  - Confidence level: {args.confidence_level}")
                    print(f"  - Method: {args.bootstrap_method}")
                    if args.adaptive_bootstrap:
                        print(f"  - Adaptive early stopping: enabled")

            # Run position bias analysis
            try:
                if args.bootstrap:
                    # Import enhanced bootstrap version from the proper package
                    try:
                        from src.statistical.position_bias import \
                            run_enhanced_position_bias_analysis
                        results = run_enhanced_position_bias_analysis(questions=questions,
                                                                      **analysis_kwargs)
                        analysis_type = "Enhanced Position Bias Analysis (with Bootstrap CIs)"
                    except ImportError as e:
                        print(f"Bootstrap functionality not available: {e}")
                        print(
                            "Make sure you've created src/statistical/bootstrap.py and updated position_bias.py")
                        logger.error("Failed to import bootstrap functionality: %s", str(e))
                        return 1
                else:
                    # Use standard analysis from the proper package
                    results = run_position_bias_analysis(questions=questions, **analysis_kwargs)
                    analysis_type = "Standard Position Bias Analysis"

                # Display results
                print(f"\n🔍 {analysis_type} Results")
                print(f"═══════════════════════════════════════════")
                print(f"Dataset: {len(questions)} questions")
                print(f"Position Frequencies: {results['position_frequencies']}")
                print(
                    f"Chi-square statistic: {results['chi_square_results']['chi_square_statistic']:.4f}")
                print(f"P-value: {results['chi_square_results']['p_value']:.6f}")
                print(
                    f"Significant bias detected: {'YES' if results['chi_square_results']['significant'] else 'NO'}")
                print(f"Predictive questions found: {len(results['predictive_questions'])}")

                # Display bootstrap results if available
                if args.bootstrap and 'bootstrap_chi_square' in results and results[
                    'bootstrap_chi_square']:
                    bootstrap_stats = results['bootstrap_chi_square']
                    ci_lower, ci_upper = bootstrap_stats['confidence_interval']
                    print(f"\n📊 Bootstrap Confidence Intervals:")
                    print(f"Chi-square statistic: {bootstrap_stats['statistic']:.4f}")
                    print(
                        f"{bootstrap_stats['confidence_level'] * 100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                    print(f"Bootstrap iterations: {bootstrap_stats['n_bootstrap']}")
                    print(f"Method: {bootstrap_stats['method']}")

                    if 'bootstrap_position_proportions' in results and results[
                        'bootstrap_position_proportions']:
                        print(f"\nPosition Proportion Bootstrap CIs:")
                        for position, prop_stats in results[
                            'bootstrap_position_proportions'].items():
                            prop_ci_lower, prop_ci_upper = prop_stats['confidence_interval']
                            print(f"  Position {position}: {prop_stats['statistic']:.3f} "
                                  f"[{prop_ci_lower:.3f}, {prop_ci_upper:.3f}]")

                    if 'bootstrap_performance' in results:
                        perf = results['bootstrap_performance']
                        print(f"\n⚡ Bootstrap Performance:")
                        print(f"Runtime: {perf['runtime_seconds']:.3f}s")
                        print(f"Total iterations: {perf['total_bootstrap_iterations']}")
                        if perf.get('convergence_achieved'):
                            print(f"Early convergence: achieved")

                print(
                    f"Position swap variants generated: {results['summary_statistics']['total_variants_generated']}")

                if args.verbose and results['predictive_questions']:
                    print(f"\nPredictive Question IDs (first 10):")
                    for qid in results['predictive_questions'][:10]:
                        print(f"  - {qid}")

                if args.output:
                    print(f"\n💾 Detailed results saved to: {args.output}")

                logger.info(
                    "Position bias analysis completed: bias_detected=%s, predictive_questions=%d, bootstrap_enabled=%s",
                    results['chi_square_results']['significant'],
                    len(results['predictive_questions']),
                    args.bootstrap
                )

                return 0

            except Exception as e:
                print(f"Error during position bias analysis: {e}")
                logger.error("Error during position bias analysis: %s", str(e), exc_info=True)
                return 1

        elif args.command == "bootstrap-test":
            print(f"🧪 Bootstrap CI Test")
            print(f"==================")

            # Generate test data
            np.random.seed(42)
            if args.data_type == "normal":
                data = np.random.normal(100, 15, args.n_samples)
                true_param = 100  # true mean
            elif args.data_type == "uniform":
                data = np.random.uniform(0, 10, args.n_samples)
                true_param = 5  # true mean
            elif args.data_type == "binary":
                data = np.random.binomial(1, 0.3, args.n_samples)
                true_param = 0.3  # true proportion

            # Select statistic function
            if args.statistic == "mean":
                stat_func = np.mean
            elif args.statistic == "median":
                stat_func = np.median
            elif args.statistic == "std":
                stat_func = np.std

            print(f"Data: {args.data_type} distribution, {args.n_samples} samples")
            print(f"Statistic: {args.statistic}")
            print(f"True parameter: {true_param}")

            if args.verbose:
                print(f"Bootstrap parameters:")
                print(f"  - Iterations: {args.n_bootstrap}")
                print(f"  - Confidence level: {args.confidence_level}")
                print(f"  - Method: {args.method}")

            # Run bootstrap
            try:
                from src.statistical.bootstrap import bootstrap_ci

                result = bootstrap_ci(
                    data,
                    stat_func,
                    n_bootstrap=args.n_bootstrap,
                    confidence_level=args.confidence_level,
                    method=args.method,
                    random_seed=42,
                    adaptive=True if args.verbose else False  # Enable adaptive mode in verbose
                )

                print(f"\n📊 Bootstrap Results:")
                print(f"Sample statistic: {result.statistic:.4f}")
                ci_lower, ci_upper = result.confidence_interval
                print(f"{args.confidence_level * 100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                print(f"CI width: {ci_upper - ci_lower:.4f}")
                print(f"Bootstrap iterations: {result.n_iterations}")
                print(f"Runtime: {result.runtime_seconds:.3f}s")
                print(f"Method: {result.method}")

                if result.converged:
                    print(f"Converged at iteration: {result.convergence_iteration}")
                elif args.verbose:
                    print(f"Did not converge (ran full {result.n_iterations} iterations)")

                # Check if true parameter is in CI
                contains_true = ci_lower <= true_param <= ci_upper
                print(f"CI contains true parameter: {'YES' if contains_true else 'NO'}")

                if args.verbose:
                    print(f"\n📈 Additional Statistics:")
                    print(f"Bootstrap estimates mean: {np.mean(result.bootstrap_estimates):.4f}")
                    print(f"Bootstrap estimates std: {np.std(result.bootstrap_estimates):.4f}")
                    print(
                        f"Theoretical CI width check: {'PASS' if ci_upper > ci_lower else 'FAIL'}")

                logger.info("Bootstrap test completed successfully")
                return 0

            except Exception as e:
                print(f"Error in bootstrap test: {e}")
                logger.error("Bootstrap test error: %s", str(e), exc_info=True)
                return 1

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
