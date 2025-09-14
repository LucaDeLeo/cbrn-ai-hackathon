#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Final Verification
# This script performs final verification of all pipeline outputs

set -euo pipefail

# Configuration with defaults
RESULTS_DIR="${RESULTS_DIR:-artifacts/results}"
FIGURES_DIR="${FIGURES_DIR:-artifacts/figs}"
REPORT_DIR="${REPORT_DIR:-docs/results}"
LOGS_DIR="${LOGS_DIR:-logs}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SUMMARY_FILE="${SUMMARY_FILE:-artifacts/results/summary.json}"
CSV_FILE="${CSV_FILE:-artifacts/results/all_results.csv}"
REPORT_FILE="${REPORT_FILE:-docs/results/report.md}"

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
    log "ERROR" "Script failed at line $line_number with exit code $exit_code"
    exit $exit_code
}

trap 'handle_error $LINENO' ERR

log "INFO" "Starting final verification"

# Verification counters
local checks_passed=0
local checks_total=0
local critical_failures=0

# Verification function
verify_check() {
    local check_name="$1"
    local check_command="$2"
    local is_critical="${3:-true}"
    
    ((checks_total++))
    log "INFO" "Verifying: $check_name"
    
    if eval "$check_command" >/dev/null 2>&1; then
        log "INFO" "✅ $check_name - PASSED"
        ((checks_passed++))
        return 0
    else
        log "ERROR" "❌ $check_name - FAILED"
        if [ "$is_critical" = "true" ]; then
            ((critical_failures++))
        fi
        return 1
    fi
}

# Verification function with custom message
verify_check_with_message() {
    local check_name="$1"
    local check_command="$2"
    local success_message="$3"
    local failure_message="$4"
    local is_critical="${5:-true}"
    
    ((checks_total++))
    log "INFO" "Verifying: $check_name"
    
    if eval "$check_command" >/dev/null 2>&1; then
        log "INFO" "✅ $check_name - PASSED: $success_message"
        ((checks_passed++))
        return 0
    else
        log "ERROR" "❌ $check_name - FAILED: $failure_message"
        if [ "$is_critical" = "true" ]; then
            ((critical_failures++))
        fi
        return 1
    fi
}

log "INFO" "=== Final Verification ==="

# Check 1: Complete artifacts/results/summary.json with all metrics
verify_check_with_message \
    "Summary file exists" \
    "[ -f '$SUMMARY_FILE' ]" \
    "Summary file found" \
    "Summary file not found" \
    "true"

if [ -f "$SUMMARY_FILE" ]; then
    verify_check_with_message \
        "Summary contains accuracy metrics" \
        "grep -q 'mcq_full_accuracy\\|choices_only_accuracy\\|cloze_full_accuracy\\|task_accuracy' '$SUMMARY_FILE'" \
        "Accuracy metrics present" \
        "Missing accuracy metrics" \
        "true"
    
    verify_check_with_message \
        "Summary contains consensus detection results" \
        "grep -q 'exploitable_fraction\\|consensus_detection\\|exploitable_count' '$SUMMARY_FILE'" \
        "Consensus detection results present" \
        "Missing consensus detection results" \
        "true"
    
    verify_check_with_message \
        "Summary contains robustness metrics" \
        "grep -q 'paraphrase_consistency\\|perturbation_fragility\\|mcq_vs_cloze' '$SUMMARY_FILE'" \
        "Robustness metrics present" \
        "Missing robustness metrics" \
        "false"
fi

# Check 2: Generated figures in artifacts/figs/
verify_check_with_message \
    "Figures directory exists" \
    "[ -d '$FIGURES_DIR' ]" \
    "Figures directory found" \
    "Figures directory not found" \
    "true"

if [ -d "$FIGURES_DIR" ]; then
    local expected_figures=(
        "paraphrase_consistency.png"
        "perturbation_fragility.png"
        "mcq_cloze_delta.png"
        "exploitable_fraction.png"
    )
    
    local figures_found=0
    for figure in "${expected_figures[@]}"; do
        if [ -f "$FIGURES_DIR/$figure" ]; then
            ((figures_found++))
        fi
    done
    
    verify_check_with_message \
        "All expected figures generated" \
        "[ $figures_found -eq ${#expected_figures[@]} ]" \
        "All $figures_found figures generated" \
        "Only $figures_found/${#expected_figures[@]} figures generated" \
        "false"
    
    if [ $figures_found -lt ${#expected_figures[@]} ]; then
        log "WARN" "Missing figures:"
        for figure in "${expected_figures[@]}"; do
            if [ ! -f "$FIGURES_DIR/$figure" ]; then
                log "WARN" "  - $figure"
            fi
        done
    fi
fi

# Check 3: Populated docs/results/report.md with real data
verify_check_with_message \
    "Report file exists" \
    "[ -f '$REPORT_FILE' ]" \
    "Report file found" \
    "Report file not found" \
    "true"

if [ -f "$REPORT_FILE" ]; then
    verify_check_with_message \
        "Report populated with real data" \
        "! grep -q '{{' '$REPORT_FILE'" \
        "No placeholders found" \
        "Report still contains placeholders" \
        "false"
    
    verify_check_with_message \
        "Report contains key metrics" \
        "grep -q 'MCQ Full\\|Choices Only\\|Cloze Full\\|Exploitable Questions' '$REPORT_FILE'" \
        "Key metrics present" \
        "Missing key metrics" \
        "true"
    
    local report_lines=$(wc -l < "$REPORT_FILE" 2>/dev/null || echo 0)
    verify_check_with_message \
        "Report has sufficient content" \
        "[ $report_lines -gt 50 ]" \
        "Report has $report_lines lines" \
        "Report has only $report_lines lines" \
        "false"
fi

# Check 4: CSV file with all results
verify_check_with_message \
    "CSV file exists" \
    "[ -f '$CSV_FILE' ]" \
    "CSV file found" \
    "CSV file not found" \
    "false"

if [ -f "$CSV_FILE" ]; then
    local csv_lines=$(wc -l < "$CSV_FILE" 2>/dev/null || echo 0)
    verify_check_with_message \
        "CSV file has data" \
        "[ $csv_lines -gt 1 ]" \
        "CSV has $csv_lines lines" \
        "CSV has only $csv_lines lines" \
        "false"
fi

# Check 5: Log files exist
verify_check_with_message \
    "Logs directory exists" \
    "[ -d '$LOGS_DIR' ]" \
    "Logs directory found" \
    "Logs directory not found" \
    "false"

if [ -d "$LOGS_DIR" ]; then
    local log_files=$(find "$LOGS_DIR" -name "*.jsonl" | wc -l)
    verify_check_with_message \
        "Log files exist" \
        "[ $log_files -gt 0 ]" \
        "Found $log_files log files" \
        "No log files found" \
        "false"
fi

# Check 6: JSON validity
if [ -f "$SUMMARY_FILE" ]; then
    verify_check_with_message \
        "Summary JSON is valid" \
        "python -c 'import json; json.load(open(\"$SUMMARY_FILE\"))'" \
        "JSON is valid" \
        "JSON is invalid" \
        "true"
fi

# Check 7: File permissions and accessibility
verify_check_with_message \
        "Results directory is accessible" \
        "[ -r '$RESULTS_DIR' ]" \
        "Results directory is readable" \
        "Results directory is not readable" \
        "true"

# Check 8: Reproducibility indicators
verify_check_with_message \
        "Consensus K is set" \
        "grep -q '\"consensus_k\"' '$SUMMARY_FILE'" \
        "Consensus K is configured" \
        "Consensus K is not configured" \
        "false"

# Summary
log "INFO" "=== Verification Summary ==="
log "INFO" "Total checks: $checks_total"
log "INFO" "Checks passed: $checks_passed"
log "INFO" "Checks failed: $((checks_total - checks_passed))"
log "INFO" "Critical failures: $critical_failures"

# Calculate success rate
local success_rate=$(python -c "print(f'{($checks_passed / $checks_total * 100):.1f}%')" 2>/dev/null || echo "N/A")
log "INFO" "Success rate: $success_rate"

# Final status
if [ $critical_failures -eq 0 ]; then
    if [ $checks_passed -eq $checks_total ]; then
        log "INFO" "🎉 All verifications passed! Pipeline completed successfully."
        exit 0
    else
        log "WARN" "⚠️  Some non-critical verifications failed, but pipeline completed."
        exit 0
    fi
else
    log "ERROR" "💥 Critical verifications failed! Pipeline has issues that need attention."
    exit 1
fi
