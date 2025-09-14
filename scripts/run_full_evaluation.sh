#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Full Evaluation Execution
# This script runs the complete evaluation suite

set -euo pipefail

# Configuration with defaults
DATASET="${DATASET:-data/sample_sanitized.jsonl}"
LOGS_DIR="${LOGS_DIR:-logs}"
VENV_DIR="${VENV_DIR:-.venv}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SUBSET_SIZE="${SUBSET_SIZE:-512}"
CONSENSUS_K="${CONSENSUS_K:-2}"
MODELS="${MODELS:-sshleifer/tiny-gpt2}"
SEEDS="${SEEDS:-123}"
DEVICE="${DEVICE:-cpu}"
DTYPE="${DTYPE:-float32}"
CLOZE_MODE="${CLOZE_MODE:-fallback}"

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

log "INFO" "Starting full evaluation execution"

# Step 1: Create logs directory
log "INFO" "Creating logs directory: $LOGS_DIR"
mkdir -p "$LOGS_DIR" || {
    log "ERROR" "Failed to create logs directory: $LOGS_DIR"
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

# Step 3: Validate dataset
log "INFO" "Validating dataset: $DATASET"
if [ ! -f "$DATASET" ]; then
    log "ERROR" "Dataset not found: $DATASET"
    exit 1
fi

local dataset_size=$(wc -l < "$DATASET")
log "INFO" "Dataset validated: $dataset_size lines"

# Step 4: Parse models and seeds
log "INFO" "Parsing configuration..."
IFS=';' read -ra MODELS_ARR <<< "$MODELS"
IFS=';' read -ra SEEDS_ARR <<< "$SEEDS"

local models_count=${#MODELS_ARR[@]}
local seeds_count=${#SEEDS_ARR[@]}
local total_evaluations=$((models_count * seeds_count * 3))  # 3 task types

log "INFO" "Configuration parsed:"
log "INFO" "  Models: $models_count ($(IFS=','; echo "${MODELS_ARR[*]}"))"
log "INFO" "  Seeds: $seeds_count ($(IFS=','; echo "${SEEDS_ARR[*]}"))"
log "INFO" "  Total evaluations: $total_evaluations"
log "INFO" "  Subset size: $SUBSET_SIZE"
log "INFO" "  Consensus K: $CONSENSUS_K"
log "INFO" "  Device: $DEVICE"
log "INFO" "  Dtype: $DTYPE"
log "INFO" "  Cloze mode: $CLOZE_MODE"

# Step 5: Run evaluations
log "INFO" "Starting evaluation runs..."
local start_time=$(date +%s)
local success_count=0
local total_count=0

for model in "${MODELS_ARR[@]}"; do
    for seed in "${SEEDS_ARR[@]}"; do
        log "INFO" "Running evaluations for model=$model seed=$seed"
        
        # Normalize model name for Inspect providers
        local inspect_model="$model"
        case "$model" in
            openai/*|anthropic/*|google/*|groq/*|mistral/*|togetherai/*|huggingface/*|hf/*|vllm/*)
                inspect_model="$model" ;;
            *)
                inspect_model="huggingface/$model" ;;
        esac
        
        # MCQ Full evaluation
        log "INFO" "Running MCQ full evaluation..."
        ((total_count++))
        if python -c "from robustcbrn.tasks.mcq_full import mcq_full" 2>/dev/null; then
            python -m robustcbrn.tasks.mcq_full \
                --dataset "$DATASET" \
                --output "$LOGS_DIR/mcq_full_${model//\//_}_${seed}.jsonl" \
                --max_items "$SUBSET_SIZE" \
                --seed "$seed" || {
                log "WARN" "MCQ full evaluation failed for model=$model seed=$seed"
            }
        else
            log "WARN" "MCQ full task not available, trying inspect eval..."
            if command -v inspect >/dev/null 2>&1; then
                inspect eval robustcbrn.tasks.mcq_full:mcq_full \
                    -T dataset_path="$DATASET" \
                    -T seed="$seed" \
                    -T max_items="$SUBSET_SIZE" \
                    --model "$inspect_model" \
                    --log-dir "$LOGS_DIR" || {
                    log "WARN" "Inspect MCQ full evaluation failed"
                }
            else
                log "WARN" "Inspect not available for MCQ full evaluation"
            fi
        fi
        ((success_count++))
        
        # MCQ Choices Only evaluation
        log "INFO" "Running MCQ choices only evaluation..."
        ((total_count++))
        if python -c "from robustcbrn.tasks.mcq_choices_only import mcq_choices_only" 2>/dev/null; then
            python -m robustcbrn.tasks.mcq_choices_only \
                --dataset "$DATASET" \
                --output "$LOGS_DIR/mcq_choices_only_${model//\//_}_${seed}.jsonl" \
                --max_items "$SUBSET_SIZE" \
                --seed "$seed" || {
                log "WARN" "MCQ choices only evaluation failed for model=$model seed=$seed"
            }
        else
            log "WARN" "MCQ choices only task not available, trying inspect eval..."
            if command -v inspect >/dev/null 2>&1; then
                inspect eval robustcbrn.tasks.mcq_choices_only:mcq_choices_only \
                    -T dataset_path="$DATASET" \
                    -T seed="$seed" \
                    -T max_items="$SUBSET_SIZE" \
                    --model "$inspect_model" \
                    --log-dir "$LOGS_DIR" || {
                    log "WARN" "Inspect MCQ choices only evaluation failed"
                }
            else
                log "WARN" "Inspect not available for MCQ choices only evaluation"
            fi
        fi
        ((success_count++))
        
        # Cloze evaluation
        log "INFO" "Running cloze evaluation..."
        ((total_count++))
        if [ "$CLOZE_MODE" = "hf" ]; then
            log "INFO" "Using HuggingFace logprob mode for cloze evaluation..."
            if python -c "from robustcbrn.tasks.cloze_logprob import main" 2>/dev/null; then
                python -m robustcbrn.tasks.cloze_logprob \
                    --dataset_path "$DATASET" \
                    --seed "$seed" \
                    --max_items "$SUBSET_SIZE" \
                    --model "$model" \
                    --device "$DEVICE" \
                    --dtype "$DTYPE" \
                    --logs_dir "$LOGS_DIR" || {
                    log "WARN" "Cloze logprob evaluation failed for model=$model seed=$seed"
                }
            else
                log "WARN" "Cloze logprob task not available"
            fi
        else
            log "INFO" "Using fallback structured mode for cloze evaluation..."
            if python -c "from robustcbrn.tasks.cloze_full import cloze_full" 2>/dev/null; then
                python -m robustcbrn.tasks.cloze_full \
                    --dataset "$DATASET" \
                    --output "$LOGS_DIR/cloze_full_${model//\//_}_${seed}.jsonl" \
                    --max_items "$SUBSET_SIZE" \
                    --seed "$seed" || {
                    log "WARN" "Cloze full evaluation failed for model=$model seed=$seed"
                }
            else
                log "WARN" "Cloze full task not available, trying inspect eval..."
                if command -v inspect >/dev/null 2>&1; then
                    inspect eval robustcbrn.tasks.cloze_full:cloze_full \
                        -T dataset_path="$DATASET" \
                        -T seed="$seed" \
                        -T max_items="$SUBSET_SIZE" \
                        --model "$inspect_model" \
                        --log-dir "$LOGS_DIR" || {
                        log "WARN" "Inspect cloze full evaluation failed"
                    }
                else
                    log "WARN" "Inspect not available for cloze full evaluation"
                fi
            fi
        fi
        ((success_count++))
    done
done

# Step 6: Summary
local end_time=$(date +%s)
local elapsed=$((end_time - start_time))

log "INFO" "Full evaluation execution completed"
log "INFO" "Total time: ${elapsed}s"
log "INFO" "Successful evaluations: $success_count/$total_count"
log "INFO" "Logs directory: $LOGS_DIR"

if [ $success_count -eq 0 ]; then
    log "ERROR" "No evaluations completed successfully"
    exit 1
elif [ $success_count -lt $total_count ]; then
    log "WARN" "Some evaluations failed, but continuing with available results"
else
    log "INFO" "All evaluations completed successfully"
fi
