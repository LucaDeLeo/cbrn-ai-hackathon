#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Results Aggregation
# This script aggregates results from multiple evaluations

set -euo pipefail

# Configuration with defaults
LOGS_DIR="${LOGS_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-artifacts/results}"
CONSENSUS_K="${CONSENSUS_K:-2}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

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

log "INFO" "Starting results aggregation"

# Step 1: Create results directory
log "INFO" "Creating results directory: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR" || {
    log "ERROR" "Failed to create results directory: $RESULTS_DIR"
    exit 1
}

# Step 2: Activate virtual environment
log "INFO" "Activating virtual environment: $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate" || {
        log "ERROR" "Failed to activate virtual environment"
        exit 1
    }
    log "INFO" "Virtual environment activated"
else
    log "ERROR" "Virtual environment not found: $VENV_DIR"
    exit 1
fi

# Step 3: Check for log files
log "INFO" "Checking for log files in: $LOGS_DIR"
if [ ! -d "$LOGS_DIR" ]; then
    log "ERROR" "Logs directory not found: $LOGS_DIR"
    exit 1
fi

local log_files=$(find "$LOGS_DIR" -name "*.jsonl" | wc -l)
log "INFO" "Found $log_files log files"

if [ $log_files -eq 0 ]; then
    log "ERROR" "No log files found in $LOGS_DIR"
    exit 1
fi

# Step 4: Run aggregation
log "INFO" "Running aggregation with consensus K=$CONSENSUS_K"

# Try using the aggregate module directly
if python -c "from robustcbrn.analysis.aggregate import main" 2>/dev/null; then
    log "INFO" "Using robustcbrn.analysis.aggregate module..."
    python -m robustcbrn.analysis.aggregate \
        --logs "$LOGS_DIR" \
        --out "$RESULTS_DIR" \
        --k "$CONSENSUS_K" || {
        log "ERROR" "Aggregation failed"
        exit 1
    }
else
    log "WARN" "Aggregate module not available, trying alternative approach..."
    
    # Create a simple aggregation script
    cat > "$RESULTS_DIR/aggregate_manual.py" << 'EOF'
#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

def aggregate_results(logs_dir, output_dir, consensus_k=2):
    """Manual aggregation of results"""
    logs_path = Path(logs_dir)
    output_path = Path(output_dir)
    
    # Find all JSONL files
    jsonl_files = list(logs_path.glob("*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {logs_dir}")
        return False
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Load all results
    all_results = []
    for file_path in jsonl_files:
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        data['source_file'] = file_path.name
                        all_results.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if not all_results:
        print("No valid results found")
        return False
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    print(f"Loaded {len(df)} results")
    
    # Basic aggregation
    summary = {
        "total_results": len(df),
        "unique_models": df['model'].nunique() if 'model' in df.columns else 0,
        "unique_seeds": df['seed'].nunique() if 'seed' in df.columns else 0,
        "tasks": df['task'].unique().tolist() if 'task' in df.columns else [],
        "consensus_k": consensus_k
    }
    
    # Calculate accuracy by task
    if 'correct' in df.columns and 'task' in df.columns:
        task_accuracy = df.groupby('task')['correct'].agg(['mean', 'count', 'std']).to_dict()
        summary['task_accuracy'] = task_accuracy
    
    # Calculate exploitable fraction for choices-only
    if 'exploitable' in df.columns and 'task' in df.columns:
        choices_df = df[df['task'] == 'mcq_choices_only']
        if not choices_df.empty and 'id' in choices_df.columns:
            # One row per item id
            item_exploitable = choices_df.groupby('id')['exploitable'].first()
            summary['exploitable_fraction'] = float(item_exploitable.mean())
            summary['exploitable_count'] = int(item_exploitable.sum())
            summary['total_items'] = len(item_exploitable)
    
    # Save summary
    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Save all results as CSV
    csv_file = output_path / "all_results.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"Summary saved to {summary_file}")
    print(f"All results saved to {csv_file}")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python aggregate_manual.py <logs_dir> <output_dir> <consensus_k>")
        sys.exit(1)
    
    logs_dir = sys.argv[1]
    output_dir = sys.argv[2]
    consensus_k = int(sys.argv[3])
    
    success = aggregate_results(logs_dir, output_dir, consensus_k)
    sys.exit(0 if success else 1)
EOF

    python "$RESULTS_DIR/aggregate_manual.py" "$LOGS_DIR" "$RESULTS_DIR" "$CONSENSUS_K" || {
        log "ERROR" "Manual aggregation failed"
        exit 1
    }
    
    # Clean up temporary script
    rm "$RESULTS_DIR/aggregate_manual.py"
fi

# Step 5: Validate outputs
log "INFO" "Validating aggregation outputs..."

local summary_file="$RESULTS_DIR/summary.json"
local csv_file="$RESULTS_DIR/all_results.csv"

if [ -f "$summary_file" ]; then
    log "INFO" "Summary file created: $summary_file"
    log "DEBUG" "Summary contents:"
    cat "$summary_file" | while read -r line; do
        log "DEBUG" "  $line"
    done
else
    log "ERROR" "Summary file not created: $summary_file"
    exit 1
fi

if [ -f "$csv_file" ]; then
    local csv_lines=$(wc -l < "$csv_file")
    log "INFO" "CSV file created: $csv_file ($csv_lines lines)"
else
    log "WARN" "CSV file not created: $csv_file"
fi

# Step 6: Summary
log "INFO" "Results aggregation completed successfully"
log "INFO" "Results directory: $RESULTS_DIR"
log "INFO" "Summary file: $summary_file"
log "INFO" "CSV file: $csv_file"
log "INFO" "Consensus K: $CONSENSUS_K"
log "INFO" "Log files processed: $log_files"
