#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Main Wrapper Script
# This script orchestrates the complete evaluation pipeline

set -euo pipefail

# Configuration with defaults
SCRIPT_DIR="${SCRIPT_DIR:-$(dirname "$0")}"
VENV_DIR="${VENV_DIR:-.venv}"
DATASET="${DATASET:-data/sample_sanitized.jsonl}"
LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-artifacts/results}"
FIGURES_DIR="${FIGURES_DIR:-artifacts/figs}"
REPORT_DIR="${REPORT_DIR:-docs/results}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SUBSET_SIZE="${SUBSET_SIZE:-512}"
CONSENSUS_K="${CONSENSUS_K:-2}"
MODELS="${MODELS:-sshleifer/tiny-gpt2}"
SEEDS="${SEEDS:-123}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
CLOZE_MODE="${CLOZE_MODE:-fallback}"

# Pipeline steps
STEPS="${STEPS:-validate,setup,discover,sample,full,aggregate,figures,tests,report,verify}"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >&2
}

# Error handling
handle_error() {
    local exit_code=$?
    local line_number=$1
    log "ERROR" "Pipeline failed at line $line_number with exit code $exit_code"
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

# Help function
show_help() {
    cat << EOF
RobustCBRN Evaluation Pipeline

Usage: $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -s, --steps STEPS       Comma-separated list of steps to run
    -d, --dataset DATASET   Dataset file path (default: $DATASET)
    -l, --logs LOGS_DIR     Logs directory (default: $LOGS_DIR)
    -r, --results RESULTS_DIR Results directory (default: $RESULTS_DIR)
    -f, --figures FIGURES_DIR Figures directory (default: $FIGURES_DIR)
    -p, --report REPORT_DIR Report directory (default: $REPORT_DIR)
    -v, --venv VENV_DIR     Virtual environment directory (default: $VENV_DIR)
    -n, --subset SUBSET_SIZE Subset size for evaluation (default: $SUBSET_SIZE)
    -k, --consensus CONSENSUS_K Consensus K value (default: $CONSENSUS_K)
    -m, --models MODELS     Semicolon-separated list of models (default: $MODELS)
    -e, --seeds SEEDS       Semicolon-separated list of seeds (default: $SEEDS)
    --device DEVICE         Device for evaluation (default: $DEVICE)
    --dtype DTYPE           Data type for evaluation (default: $DTYPE)
    --cloze-mode MODE       Cloze evaluation mode (default: $CLOZE_MODE)
    --log-level LEVEL       Log level (default: $LOG_LEVEL)

AVAILABLE STEPS:
    validate    - Platform detection and dependency validation
    setup       - Environment setup and dependency installation
    discover    - Discover and validate entry points
    sample      - Run sample evaluation
    full        - Run full evaluation suite
    aggregate   - Aggregate results from evaluations
    figures     - Generate figures and visualizations
    tests       - Run tests and security validation
    report      - Generate final report
    verify      - Final verification of all outputs

EXAMPLES:
    # Run complete pipeline
    $0

    # Run only setup and sample evaluation
    $0 --steps setup,sample

    # Run with custom dataset and subset size
    $0 --dataset data/custom.jsonl --subset 256

    # Run with different models and seeds
    $0 --models "model1;model2" --seeds "123;456;789"

ENVIRONMENT VARIABLES:
    All options can also be set via environment variables with the same names.

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -s|--steps)
            STEPS="$2"
            shift 2
            ;;
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -l|--logs)
            LOGS_DIR="$2"
            shift 2
            ;;
        -r|--results)
            RESULTS_DIR="$2"
            shift 2
            ;;
        -f|--figures)
            FIGURES_DIR="$2"
            shift 2
            ;;
        -p|--report)
            REPORT_DIR="$2"
            shift 2
            ;;
        -v|--venv)
            VENV_DIR="$2"
            shift 2
            ;;
        -n|--subset)
            SUBSET_SIZE="$2"
            shift 2
            ;;
        -k|--consensus)
            CONSENSUS_K="$2"
            shift 2
            ;;
        -m|--models)
            MODELS="$2"
            shift 2
            ;;
        -e|--seeds)
            SEEDS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --cloze-mode)
            CLOZE_MODE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        *)
            log "ERROR" "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Export configuration for sub-scripts
export VENV_DIR DATASET LOGS_DIR RESULTS_DIR FIGURES_DIR REPORT_DIR LOG_LEVEL
export SUBSET_SIZE CONSENSUS_K MODELS SEEDS DEVICE DTYPE CLOZE_MODE

log "INFO" "Starting RobustCBRN Evaluation Pipeline"
log "INFO" "Configuration:"
log "INFO" "  Script directory: $SCRIPT_DIR"
log "INFO" "  Virtual environment: $VENV_DIR"
log "INFO" "  Dataset: $DATASET"
log "INFO" "  Logs directory: $LOGS_DIR"
log "INFO" "  Results directory: $RESULTS_DIR"
log "INFO" "  Figures directory: $FIGURES_DIR"
log "INFO" "  Report directory: $REPORT_DIR"
log "INFO" "  Subset size: $SUBSET_SIZE"
log "INFO" "  Consensus K: $CONSENSUS_K"
log "INFO" "  Models: $MODELS"
log "INFO" "  Seeds: $SEEDS"
log "INFO" "  Device: $DEVICE"
log "INFO" "  Dtype: $DTYPE"
log "INFO" "  Cloze mode: $CLOZE_MODE"
log "INFO" "  Log level: $LOG_LEVEL"
log "INFO" "  Steps: $STEPS"

# Parse steps
IFS=',' read -ra STEP_ARRAY <<< "$STEPS"
log "INFO" "Steps to execute: ${#STEP_ARRAY[@]} (${STEP_ARRAY[*]})"

# Step execution function
execute_step() {
    local step="$1"
    local script_file=""
    local step_name=""
    
    case "$step" in
        validate)
            script_file="$SCRIPT_DIR/validate_platform.sh"
            step_name="Platform Detection and Dependency Validation"
            ;;
        setup)
            script_file="$SCRIPT_DIR/setup_env.sh"
            step_name="Environment Setup"
            ;;
        discover)
            script_file="$SCRIPT_DIR/discover_entry_points.sh"
            step_name="Entry Point Discovery"
            ;;
        sample)
            script_file="$SCRIPT_DIR/run_sample_evaluation.sh"
            step_name="Sample Evaluation"
            ;;
        full)
            script_file="$SCRIPT_DIR/run_full_evaluation.sh"
            step_name="Full Evaluation"
            ;;
        aggregate)
            script_file="$SCRIPT_DIR/aggregate_results.sh"
            step_name="Results Aggregation"
            ;;
        figures)
            script_file="$SCRIPT_DIR/generate_figures.sh"
            step_name="Figure Generation"
            ;;
        tests)
            script_file="$SCRIPT_DIR/run_tests_and_security.sh"
            step_name="Testing and Security"
            ;;
        report)
            script_file="$SCRIPT_DIR/generate_report.sh"
            step_name="Report Generation"
            ;;
        verify)
            script_file="$SCRIPT_DIR/final_verification.sh"
            step_name="Final Verification"
            ;;
        *)
            log "ERROR" "Unknown step: $step"
            return 1
            ;;
    esac
    
    if [ ! -f "$script_file" ]; then
        log "ERROR" "Script file not found: $script_file"
        return 1
    fi
    
    if [ ! -x "$script_file" ]; then
        log "WARN" "Script file not executable, making it executable: $script_file"
        chmod +x "$script_file"
    fi
    
    log "INFO" "Executing step: $step_name"
    log "INFO" "Script: $script_file"
    
    local start_time=$(date +%s)
    if bash "$script_file"; then
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        log "INFO" "Step completed successfully: $step_name (${elapsed}s)"
        return 0
    else
        local end_time=$(date +%s)
        local elapsed=$((end_time - start_time))
        log "ERROR" "Step failed: $step_name (${elapsed}s)"
        return 1
    fi
}

# Execute steps
local start_time=$(date +%s)
local successful_steps=0
local total_steps=${#STEP_ARRAY[@]}

for step in "${STEP_ARRAY[@]}"; do
    if execute_step "$step"; then
        ((successful_steps++))
    else
        log "ERROR" "Pipeline stopped due to step failure: $step"
        break
    fi
done

# Final summary
local end_time=$(date +%s)
local total_elapsed=$((end_time - start_time))

log "INFO" "=== Pipeline Summary ==="
log "INFO" "Total time: ${total_elapsed}s"
log "INFO" "Steps completed: $successful_steps/$total_steps"
log "INFO" "Success rate: $(python -c "print(f'{($successful_steps / $total_steps * 100):.1f}%')" 2>/dev/null || echo "N/A")"

if [ $successful_steps -eq $total_steps ]; then
    log "INFO" "🎉 Pipeline completed successfully!"
    log "INFO" "Results available in:"
    log "INFO" "  - Logs: $LOGS_DIR"
    log "INFO" "  - Results: $RESULTS_DIR"
    log "INFO" "  - Figures: $FIGURES_DIR"
    log "INFO" "  - Report: $REPORT_DIR"
    exit 0
else
    log "ERROR" "💥 Pipeline failed after $successful_steps/$total_steps steps"
    exit 1
fi
