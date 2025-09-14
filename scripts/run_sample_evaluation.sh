#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Sample Execution
# This script runs sample evaluations to validate the pipeline

set -euo pipefail

# Configuration with defaults
DATASET="${DATASET:-data/sample_sanitized.jsonl}"
LOGS_DIR="${LOGS_DIR:-logs}"
MAKEFILE="${MAKEFILE:-Makefile}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SUBSET_SIZE="${SUBSET_SIZE:-128}"

# Platform-specific configuration
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash/Cygwin)
    VENV_ACTIVATE="$VENV_DIR/Scripts/activate"
    MAKE_CMD="${MAKE_CMD:-make}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
    MAKE_CMD="${MAKE_CMD:-make}"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
    MAKE_CMD="${MAKE_CMD:-make}"
else
    # Default
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
    MAKE_CMD="${MAKE_CMD:-make}"
fi

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

log "INFO" "Starting sample evaluation execution"

# Step 1: Create logs directory
log "INFO" "Creating logs directory: $LOGS_DIR"
mkdir -p "$LOGS_DIR" || {
    log "ERROR" "Failed to create logs directory: $LOGS_DIR"
    exit 1
}

# Step 2: Check dataset availability
log "INFO" "Checking dataset availability: $DATASET"
if [ -f "$DATASET" ]; then
    local dataset_size=$(wc -l < "$DATASET")
    log "INFO" "Dataset found with $dataset_size lines"
else
    log "ERROR" "Dataset not found: $DATASET"
    log "INFO" "Looking for alternative datasets..."
    find . -name "*.jsonl" | head -5 | while read -r file; do
        log "DEBUG" "Found dataset: $file"
    done
    exit 1
fi

# Step 3: Activate virtual environment
log "INFO" "Activating virtual environment: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_ACTIVATE" || {
        log "ERROR" "Failed to activate virtual environment"
        exit 1
    }
    log "INFO" "Virtual environment activated"
else
    log "ERROR" "Virtual environment not found: $VENV_DIR"
    exit 1
fi

# Step 4: Try running sample evaluation
log "INFO" "Attempting to run sample evaluation..."

# First try: Use Makefile if available
if [ -f "$MAKEFILE" ] && grep -q "^sample:" "$MAKEFILE" && command -v "$MAKE_CMD" >/dev/null 2>&1; then
    log "INFO" "Running sample using Makefile..."
    "$MAKE_CMD" sample || {
        log "WARN" "Makefile sample target failed, trying direct execution"
    }
else
    log "INFO" "Makefile sample target not available or make not found, trying direct execution"
fi

# Second try: Direct execution
log "INFO" "Running direct sample evaluation..."

# Check if we can run MCQ full evaluation
log "INFO" "Testing MCQ full evaluation..."
if python -c "from robustcbrn.tasks.mcq_full import mcq_full" 2>/dev/null; then
    log "INFO" "Running MCQ full evaluation..."
    python -m robustcbrn.tasks.mcq_full \
        --dataset "$DATASET" \
        --output "$LOGS_DIR/sample_mcq_full.jsonl" \
        --max_items "$SUBSET_SIZE" || {
        log "WARN" "MCQ full evaluation failed"
    }
else
    log "WARN" "MCQ full task not available"
fi

# Check if we can run MCQ choices only evaluation
log "INFO" "Testing MCQ choices only evaluation..."
if python -c "from robustcbrn.tasks.mcq_choices_only import mcq_choices_only" 2>/dev/null; then
    log "INFO" "Running MCQ choices only evaluation..."
    python -m robustcbrn.tasks.mcq_choices_only \
        --dataset "$DATASET" \
        --output "$LOGS_DIR/sample_mcq_choices_only.jsonl" \
        --max_items "$SUBSET_SIZE" || {
        log "WARN" "MCQ choices only evaluation failed"
    }
else
    log "WARN" "MCQ choices only task not available"
fi

# Check if we can run cloze evaluation
log "INFO" "Testing cloze evaluation..."
if python -c "from robustcbrn.tasks.cloze_full import cloze_full" 2>/dev/null; then
    log "INFO" "Running cloze evaluation..."
    python -m robustcbrn.tasks.cloze_full \
        --dataset "$DATASET" \
        --output "$LOGS_DIR/sample_cloze_full.jsonl" \
        --max_items "$SUBSET_SIZE" || {
        log "WARN" "Cloze evaluation failed"
    }
else
    log "WARN" "Cloze task not available"
fi

# Step 5: Validate outputs
log "INFO" "Validating evaluation outputs..."
local output_files=(
    "$LOGS_DIR/sample_mcq_full.jsonl"
    "$LOGS_DIR/sample_mcq_choices_only.jsonl"
    "$LOGS_DIR/sample_cloze_full.jsonl"
)

local success_count=0
for output_file in "${output_files[@]}"; do
    if [ -f "$output_file" ]; then
        local file_size=$(wc -l < "$output_file")
        log "INFO" "Output file created: $output_file ($file_size lines)"
        ((success_count++))
    else
        log "WARN" "Output file not created: $output_file"
    fi
done

# Step 6: Summary
log "INFO" "Sample evaluation execution completed"
log "INFO" "Successful evaluations: $success_count/${#output_files[@]}"
log "INFO" "Logs directory: $LOGS_DIR"
log "INFO" "Dataset used: $DATASET"
log "INFO" "Subset size: $SUBSET_SIZE"

if [ $success_count -eq 0 ]; then
    log "ERROR" "No evaluations completed successfully"
    exit 1
elif [ $success_count -lt ${#output_files[@]} ]; then
    log "WARN" "Some evaluations failed, but continuing with available results"
else
    log "INFO" "All evaluations completed successfully"
fi
