#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Entry Point Discovery
# This script discovers and validates evaluation entry points

set -euo pipefail

# Configuration with defaults
EVALSET_SCRIPT="${EVALSET_SCRIPT:-scripts/run_evalset.sh}"
SAMPLE_SCRIPT="${SAMPLE_SCRIPT:-scripts/run_sample.sh}"
MAKEFILE="${MAKEFILE:-Makefile}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
ROBUSTCBRN_DIR="${ROBUSTCBRN_DIR:-robustcbrn}"

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

log "INFO" "Starting entry point discovery for RobustCBRN evaluation pipeline"

# Step 1: Check for run_evalset.sh script
log "INFO" "Checking for evaluation script: $EVALSET_SCRIPT"
if [ -f "$EVALSET_SCRIPT" ]; then
    log "INFO" "Found run_evalset.sh script"
    log "DEBUG" "Script contents:"
    cat "$EVALSET_SCRIPT" | while read -r line; do
        log "DEBUG" "  $line"
    done
else
    log "WARN" "run_evalset.sh not found: $EVALSET_SCRIPT"
    log "INFO" "Looking for alternative entry points..."
    
    # Check for other potential entry points
    log "DEBUG" "Searching for Python entry points..."
    find . -name "*.py" | grep -E "(run|main|eval)" | head -10 | while read -r file; do
        log "DEBUG" "Found potential entry point: $file"
    done
fi

# Step 2: Check for sample script
log "INFO" "Checking for sample script: $SAMPLE_SCRIPT"
if [ -f "$SAMPLE_SCRIPT" ]; then
    log "INFO" "Found sample script"
    log "DEBUG" "Sample script contents:"
    cat "$SAMPLE_SCRIPT" | while read -r line; do
        log "DEBUG" "  $line"
    done
else
    log "WARN" "Sample script not found: $SAMPLE_SCRIPT"
fi

# Step 3: Check Makefile targets
log "INFO" "Checking Makefile targets..."
if [ -f "$MAKEFILE" ]; then
    log "INFO" "Makefile found, checking available targets:"
    grep -E "^[a-zA-Z][a-zA-Z0-9_-]*:" "$MAKEFILE" | while read -r target; do
        log "DEBUG" "  Available target: $target"
    done
    
    # Check for specific targets we need
    if grep -q "^sample:" "$MAKEFILE"; then
        log "INFO" "Makefile has 'sample' target"
    else
        log "WARN" "Makefile missing 'sample' target"
    fi
    
    if grep -q "^run:" "$MAKEFILE"; then
        log "INFO" "Makefile has 'run' target"
    else
        log "WARN" "Makefile missing 'run' target"
    fi
else
    log "WARN" "Makefile not found: $MAKEFILE"
fi

# Step 4: Check robustcbrn package structure
log "INFO" "Checking robustcbrn package structure..."
if [ -d "$ROBUSTCBRN_DIR" ]; then
    log "INFO" "robustcbrn package directory found"
    log "DEBUG" "Package structure:"
    find "$ROBUSTCBRN_DIR" -type f -name "*.py" | head -20 | while read -r file; do
        log "DEBUG" "  $file"
    done
    
    # Check tasks directory specifically
    if [ -d "$ROBUSTCBRN_DIR/tasks" ]; then
        log "INFO" "Tasks directory found"
        log "DEBUG" "Available tasks:"
        ls -la "$ROBUSTCBRN_DIR/tasks/" | while read -r line; do
            log "DEBUG" "  $line"
        done
    else
        log "ERROR" "Tasks directory not found: $ROBUSTCBRN_DIR/tasks"
        exit 1
    fi
else
    log "ERROR" "robustcbrn package directory not found: $ROBUSTCBRN_DIR"
    exit 1
fi

# Step 5: Validate Python module imports
log "INFO" "Validating Python module imports..."
if command -v python >/dev/null 2>&1; then
    python -c "
import sys
try:
    import robustcbrn
    print('robustcbrn package import successful')
except ImportError as e:
    print(f'robustcbrn package import failed: {e}')
    sys.exit(1)

try:
    from robustcbrn.tasks import mcq_full, mcq_choices_only, cloze_full
    print('Task modules import successful')
except ImportError as e:
    print(f'Task modules import failed: {e}')
    sys.exit(1)
" || {
        log "ERROR" "Python module validation failed"
        exit 1
    }
else
    log "ERROR" "Python not available for module validation"
    exit 1
fi

log "INFO" "Entry point discovery completed successfully"
log "INFO" "Available entry points:"
log "INFO" "  - Makefile targets: $(grep -E "^[a-zA-Z][a-zA-Z0-9_-]*:" "$MAKEFILE" 2>/dev/null | wc -l || echo 0)"
log "INFO" "  - Evaluation script: $([ -f "$EVALSET_SCRIPT" ] && echo "Yes" || echo "No")"
log "INFO" "  - Sample script: $([ -f "$SAMPLE_SCRIPT" ] && echo "Yes" || echo "No")"
log "INFO" "  - Python tasks: $(find "$ROBUSTCBRN_DIR/tasks" -name "*.py" 2>/dev/null | wc -l || echo 0)"
