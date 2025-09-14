#!/usr/bin/env bash
set -euo pipefail

# Enhanced orchestrator with parallel execution for robustness checks
# - Supports parallel model evaluation
# - Includes progress monitoring
# - Handles partial failures gracefully
# - Adds performance metrics

# Load environment if exists
if [ -f .env ]; then
  set -a; source .env; set +a
fi

# Configuration
BENIGN_DATASET=${BENIGN_DATASET:-data/benign_pairs_sanitized.jsonl}
LOGS_DIR=${LOGS_DIR:-logs}
RESULTS_DIR=${RESULTS_DIR:-artifacts/results}
CONSENSUS_K=${CONSENSUS_K:-2}
MAX_PARALLEL=${MAX_PARALLEL:-4}  # Maximum parallel jobs
TIMEOUT_PER_JOB=${TIMEOUT_PER_JOB:-600}  # Timeout in seconds per job
RETRY_ATTEMPTS=${RETRY_ATTEMPTS:-2}  # Number of retry attempts

# Setup directories
mkdir -p "$LOGS_DIR" "$RESULTS_DIR"
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Parse models and seeds
IFS=';' read -ra MODELS_ARR <<< "${MODELS:-meta-llama/Llama-3.1-8B-Instruct;mistralai/Mistral-7B-Instruct-v0.3}"
IFS=';' read -ra SEEDS_ARR <<< "${SEEDS:-123;456}"
SUBSET=${SUBSET:-128}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Function to run a single evaluation job
run_evaluation_job() {
    local model="$1"
    local seed="$2"
    local job_id="$3"
    local job_file="$TEMP_DIR/job_${job_id}.status"

    log_info "Starting job $job_id: model=$model, seed=$seed"

    # Initialize job status
    echo "RUNNING" > "$job_file"

    # Run with timeout and retry
    local attempt=1
    local success=false

    while [ $attempt -le $RETRY_ATTEMPTS ] && [ "$success" = "false" ]; do
        if [ $attempt -gt 1 ]; then
            log_warn "Retry attempt $attempt for job $job_id"
        fi

        # Use gtimeout on macOS, timeout on Linux
        TIMEOUT_CMD="timeout"
        if command -v gtimeout &> /dev/null; then
            TIMEOUT_CMD="gtimeout"
        fi

        if $TIMEOUT_CMD "$TIMEOUT_PER_JOB" \
            .venv/bin/inspect eval robustcbrn.tasks.benign_policy_pairs:benign_policy_pairs \
            --arg dataset_path="$BENIGN_DATASET" \
            --arg seed="$seed" \
            --arg max_items="$SUBSET" \
            --model "$model" \
            --log-dir "$LOGS_DIR" 2>&1 | tee "$TEMP_DIR/job_${job_id}.log"; then

            success=true
            echo "SUCCESS" > "$job_file"
            log_info "Job $job_id completed successfully"
        else
            log_warn "Job $job_id failed on attempt $attempt"
            ((attempt++))
        fi
    done

    if [ "$success" = "false" ]; then
        echo "FAILED" > "$job_file"
        log_error "Job $job_id failed after $RETRY_ATTEMPTS attempts"
        return 1
    fi

    return 0
}

# Function to monitor job progress
monitor_jobs() {
    local total_jobs=$1
    local completed=0
    local failed=0

    while true; do
        completed=0
        failed=0

        for job_file in "$TEMP_DIR"/job_*.status; do
            [ -f "$job_file" ] || continue
            status=$(cat "$job_file")
            case "$status" in
                SUCCESS)
                    ((completed++))
                    ;;
                FAILED)
                    ((failed++))
                    ;;
            esac
        done

        local running=$((total_jobs - completed - failed))

        # Display progress
        printf "\r[Progress] Total: %d | Running: %d | Completed: %d | Failed: %d" \
            "$total_jobs" "$running" "$completed" "$failed"

        if [ $((completed + failed)) -eq "$total_jobs" ]; then
            echo  # New line after progress
            break
        fi

        sleep 2
    done

    return $failed
}

# Main execution
log_info "Starting robustness suite with parallel execution"
log_info "Configuration: MAX_PARALLEL=$MAX_PARALLEL, TIMEOUT=$TIMEOUT_PER_JOB, RETRIES=$RETRY_ATTEMPTS"
log_info "Dataset: $BENIGN_DATASET"
log_info "Models: ${#MODELS_ARR[@]} models, Seeds: ${#SEEDS_ARR[@]} seeds"

# Validate dataset before starting
if ! [ -f "$BENIGN_DATASET" ]; then
    log_error "Dataset file not found: $BENIGN_DATASET"
    exit 1
fi

# Validate dataset schema
log_info "Validating dataset schema..."
if .venv/bin/python -c "
from robustcbrn.utils.validation import validate_benign_pairs
try:
    validate_benign_pairs('$BENIGN_DATASET')
    print('Dataset validation passed')
except Exception as e:
    print(f'Dataset validation failed: {e}')
    exit(1)
" 2>&1; then
    log_info "Dataset validation successful"
else
    log_error "Dataset validation failed"
    exit 1
fi

# Create job queue
job_id=0
declare -a job_pids=()

# Start time for performance metrics
start_time=$(date +%s)

# Launch jobs with parallelism control
for M in "${MODELS_ARR[@]}"; do
    for S in "${SEEDS_ARR[@]}"; do
        # Wait if we've reached max parallel jobs
        while [ ${#job_pids[@]} -ge $MAX_PARALLEL ]; do
            # Check for completed jobs
            new_pids=()
            for pid in "${job_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                fi
            done
            job_pids=("${new_pids[@]}")

            if [ ${#job_pids[@]} -ge $MAX_PARALLEL ]; then
                sleep 1
            fi
        done

        # Launch job in background
        run_evaluation_job "$M" "$S" "$job_id" &
        job_pids+=($!)
        ((job_id++))
    done
done

total_jobs=$job_id
log_info "Launched $total_jobs evaluation jobs"

# Monitor progress
monitor_jobs "$total_jobs" &
monitor_pid=$!

# Wait for all jobs to complete
for pid in "${job_pids[@]}"; do
    wait "$pid" || true
done

# Stop monitor
kill $monitor_pid 2>/dev/null || true
wait $monitor_pid 2>/dev/null || true

# Calculate execution time
end_time=$(date +%s)
duration=$((end_time - start_time))
log_info "All jobs completed in ${duration} seconds"

# Count results
completed_count=$(grep -l "SUCCESS" "$TEMP_DIR"/job_*.status 2>/dev/null | wc -l)
failed_count=$(grep -l "FAILED" "$TEMP_DIR"/job_*.status 2>/dev/null | wc -l)

log_info "Final results: $completed_count successful, $failed_count failed"

# Show failed jobs if any
if [ "$failed_count" -gt 0 ]; then
    log_warn "Failed jobs:"
    for job_file in "$TEMP_DIR"/job_*.status; do
        if grep -q "FAILED" "$job_file"; then
            job_num=$(basename "$job_file" | sed 's/job_\(.*\)\.status/\1/')
            log_warn "  - Job $job_num (see $TEMP_DIR/job_${job_num}.log for details)"
        fi
    done
fi

# Run aggregation
log_info "Running aggregation..."
if .venv/bin/python -m robustcbrn.analysis.aggregate \
    --logs "$LOGS_DIR" \
    --out "$RESULTS_DIR" \
    --k "$CONSENSUS_K" 2>&1 | tee "$TEMP_DIR/aggregate.log"; then
    log_info "Aggregation completed successfully"

    # Display summary metrics if available
    if [ -f "$RESULTS_DIR/summary.json" ]; then
        log_info "Summary metrics:"
        .venv/bin/python -c "
import json
with open('$RESULTS_DIR/summary.json') as f:
    data = json.load(f)
    if 'benign_pair_stress' in data:
        metrics = data['benign_pair_stress']
        if metrics.get('present'):
            print(f'  - Compliance (safe): {metrics.get(\"compliance_safe\", 0):.2%}')
            print(f'  - Refusal (unsafe): {metrics.get(\"refusal_unsafe\", 0):.2%}')
            print(f'  - Pair consistency: {metrics.get(\"pair_consistency\", 0):.2%}')
        else:
            print('  No benign pair metrics found')
"
    fi
else
    log_error "Aggregation failed (see $TEMP_DIR/aggregate.log for details)"
    exit 1
fi

# Performance report
log_info "Performance Report:"
log_info "  - Total duration: ${duration}s"
log_info "  - Jobs per second: $(echo "scale=2; $total_jobs / $duration" | bc)"
log_info "  - Average time per job: $(echo "scale=2; $duration / $total_jobs" | bc)s"
log_info "  - Parallelism efficiency: $(echo "scale=2; 100 * $total_jobs / ($duration * $MAX_PARALLEL / 60)" | bc)%"

# Clean up temp files unless there were failures
if [ "$failed_count" -eq 0 ]; then
    rm -rf "$TEMP_DIR"
    log_info "Temporary files cleaned up"
else
    log_warn "Keeping temporary files for debugging: $TEMP_DIR"
fi

log_info "Robustness suite completed"

# Exit with appropriate code
if [ "$failed_count" -gt 0 ]; then
    exit 1
fi
