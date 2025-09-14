#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Environment Setup
# This script handles the complete environment setup for the evaluation pipeline

set -euo pipefail

# Configuration with defaults
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"
PYTHON_VERSION="${PYTHON_VERSION:-python3}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# Platform-specific configuration
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows (Git Bash/Cygwin)
    PYTHON_VERSION="${PYTHON_VERSION:-python}"
    VENV_ACTIVATE="$VENV_DIR/Scripts/activate"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    PYTHON_VERSION="${PYTHON_VERSION:-python3}"
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    PYTHON_VERSION="${PYTHON_VERSION:-python3}"
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
else
    # Default
    PYTHON_VERSION="${PYTHON_VERSION:-python3}"
    VENV_ACTIVATE="$VENV_DIR/bin/activate"
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

log "INFO" "Starting environment setup for RobustCBRN evaluation pipeline"

# Step 1: Check if Makefile exists
log "INFO" "Checking for Makefile..."
if [ -f "Makefile" ]; then
    log "INFO" "Makefile found"
    log "DEBUG" "Makefile contents:"
    cat Makefile | while read -r line; do
        log "DEBUG" "  $line"
    done
else
    log "WARN" "Makefile not found - will use direct commands"
fi

# Step 2: Check if requirements.txt exists
log "INFO" "Checking for requirements file..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    log "INFO" "Requirements file found: $REQUIREMENTS_FILE"
    log "DEBUG" "Requirements contents:"
    cat "$REQUIREMENTS_FILE" | while read -r line; do
        log "DEBUG" "  $line"
    done
else
    log "WARN" "Requirements file not found: $REQUIREMENTS_FILE"
fi

# Step 3: Check Python version and availability
log "INFO" "Checking Python installation..."
if command -v "$PYTHON_VERSION" >/dev/null 2>&1; then
    local python_ver=$("$PYTHON_VERSION" --version 2>&1)
    log "INFO" "Python found: $python_ver"
else
    log "ERROR" "Python not found: $PYTHON_VERSION"
    exit 1
fi

# Step 4: Check for existing virtual environment
log "INFO" "Checking for virtual environment..."
if [ -d "$VENV_DIR" ]; then
    log "INFO" "Virtual environment already exists: $VENV_DIR"
    log "INFO" "Activating existing virtual environment..."
    source "$VENV_ACTIVATE" || {
        log "ERROR" "Failed to activate virtual environment"
        exit 1
    }
else
    log "INFO" "Creating new virtual environment: $VENV_DIR"
    "$PYTHON_VERSION" -m venv "$VENV_DIR" || {
        log "ERROR" "Failed to create virtual environment"
        exit 1
    }
    log "INFO" "Activating virtual environment..."
    source "$VENV_ACTIVATE" || {
        log "ERROR" "Failed to activate virtual environment"
        exit 1
    }
fi

# Step 5: Upgrade pip
log "INFO" "Upgrading pip..."
pip install --upgrade pip || {
    log "ERROR" "Failed to upgrade pip"
    exit 1
}

# Step 6: Install dependencies
log "INFO" "Installing dependencies..."
if [ -f "$REQUIREMENTS_FILE" ]; then
    log "INFO" "Installing from requirements file: $REQUIREMENTS_FILE"
    pip install -r "$REQUIREMENTS_FILE" || {
        log "ERROR" "Failed to install requirements from $REQUIREMENTS_FILE"
        exit 1
    }
else
    log "WARN" "Requirements file not found. Installing common ML dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 || {
        log "WARN" "Failed to install PyTorch with CUDA support, trying CPU version"
        pip install torch torchvision torchaudio
    }
    pip install transformers datasets accelerate evaluate || {
        log "ERROR" "Failed to install transformers ecosystem"
        exit 1
    }
    pip install inspect-ai || {
        log "ERROR" "Failed to install inspect-ai"
        exit 1
    }
    pip install numpy pandas scipy scikit-learn || {
        log "ERROR" "Failed to install scientific computing packages"
        exit 1
    }
    pip install matplotlib seaborn plotly || {
        log "ERROR" "Failed to install visualization packages"
        exit 1
    }
    pip install typer rich || {
        log "ERROR" "Failed to install CLI packages"
        exit 1
    }
    pip install pytest pytest-cov || {
        log "ERROR" "Failed to install testing packages"
        exit 1
    }
fi

# Step 7: Verify GPU setup if available
log "INFO" "Verifying GPU setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA devices: {torch.cuda.device_count()}')
    print(f'Current device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
else:
    print('CUDA not available - using CPU')
" || {
    log "WARN" "GPU verification failed, but continuing..."
}

log "INFO" "Environment setup completed successfully"
log "INFO" "Virtual environment: $VENV_DIR"
log "INFO" "Python version: $($PYTHON_VERSION --version)"
log "INFO" "Pip version: $(pip --version)"
