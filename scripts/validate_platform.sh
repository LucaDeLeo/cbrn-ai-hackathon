#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Platform Detection and Dependency Validation
# This script detects the platform and validates all dependencies

set -euo pipefail

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

# Platform detection
detect_platform() {
    local os_name=""
    local os_version=""
    local shell_type=""
    local python_cmd=""
    local make_cmd=""
    
    # Detect operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        os_name="Linux"
        os_version=$(lsb_release -d 2>/dev/null | cut -f2 || uname -r)
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        os_name="macOS"
        os_version=$(sw_vers -productVersion 2>/dev/null || uname -r)
    elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
        os_name="Windows"
        os_version=$(uname -r)
    elif [[ "$OSTYPE" == "freebsd"* ]]; then
        os_name="FreeBSD"
        os_version=$(uname -r)
    else
        os_name="Unknown"
        os_version=$(uname -r)
    fi
    
    # Detect shell type
    if [[ -n "${BASH_VERSION:-}" ]]; then
        shell_type="bash"
    elif [[ -n "${ZSH_VERSION:-}" ]]; then
        shell_type="zsh"
    else
        shell_type="unknown"
    fi
    
    # Detect Python command
    if command -v python3 >/dev/null 2>&1; then
        python_cmd="python3"
    elif command -v python >/dev/null 2>&1; then
        python_cmd="python"
    else
        python_cmd=""
    fi
    
    # Detect make command
    if command -v make >/dev/null 2>&1; then
        make_cmd="make"
    elif command -v gmake >/dev/null 2>&1; then
        make_cmd="gmake"
    else
        make_cmd=""
    fi
    
    # Export platform information
    export PLATFORM_OS="$os_name"
    export PLATFORM_VERSION="$os_version"
    export PLATFORM_SHELL="$shell_type"
    export PLATFORM_PYTHON="$python_cmd"
    export PLATFORM_MAKE="$make_cmd"
    
    log "INFO" "Platform detection completed:"
    log "INFO" "  OS: $os_name ($os_version)"
    log "INFO" "  Shell: $shell_type"
    log "INFO" "  Python: $python_cmd"
    log "INFO" "  Make: $make_cmd"
}

# Dependency validation
validate_dependencies() {
    local missing_deps=()
    local optional_deps=()
    local warnings=()
    
    log "INFO" "Validating dependencies..."
    
    # Required dependencies
    local required_deps=(
        "bash:Shell interpreter"
        "python3:Python 3 interpreter"
        "pip:Python package manager"
        "git:Version control"
    )
    
    # Optional dependencies
    local optional_deps_list=(
        "make:Build tool"
        "curl:HTTP client"
        "wget:File downloader"
        "jq:JSON processor"
        "inspect:Evaluation framework"
    )
    
    # Check required dependencies
    for dep in "${required_deps[@]}"; do
        local cmd="${dep%%:*}"
        local desc="${dep##*:}"
        
        if command -v "$cmd" >/dev/null 2>&1; then
            local version=""
            case "$cmd" in
                bash)
                    version=$($cmd --version 2>&1 | head -n1 | cut -d' ' -f4)
                    ;;
                python3)
                    version=$($cmd --version 2>&1 | cut -d' ' -f2)
                    ;;
                pip)
                    version=$($cmd --version 2>&1 | cut -d' ' -f2)
                    ;;
                git)
                    version=$($cmd --version 2>&1 | cut -d' ' -f3)
                    ;;
            esac
            log "INFO" "✅ $desc: $cmd ($version)"
        else
            log "ERROR" "❌ $desc: $cmd (MISSING)"
            missing_deps+=("$cmd")
        fi
    done
    
    # Check optional dependencies
    for dep in "${optional_deps_list[@]}"; do
        local cmd="${dep%%:*}"
        local desc="${dep##*:}"
        
        if command -v "$cmd" >/dev/null 2>&1; then
            local version=""
            case "$cmd" in
                make)
                    version=$($cmd --version 2>&1 | head -n1 | cut -d' ' -f3)
                    ;;
                curl)
                    version=$($cmd --version 2>&1 | head -n1 | cut -d' ' -f2)
                    ;;
                wget)
                    version=$($cmd --version 2>&1 | head -n1 | cut -d' ' -f3)
                    ;;
                jq)
                    version=$($cmd --version 2>&1 | cut -d' ' -f2)
                    ;;
                inspect)
                    version=$($cmd --version 2>&1 | head -n1 | cut -d' ' -f2)
                    ;;
            esac
            log "INFO" "✅ $desc: $cmd ($version)"
        else
            log "WARN" "⚠️  $desc: $cmd (OPTIONAL - NOT FOUND)"
            optional_deps+=("$cmd")
        fi
    done
    
    # Python package validation
    validate_python_packages
    
    # Platform-specific validation
    validate_platform_specific
    
    # Summary
    if [ ${#missing_deps[@]} -eq 0 ]; then
        log "INFO" "✅ All required dependencies are available"
        return 0
    else
        log "ERROR" "❌ Missing required dependencies: ${missing_deps[*]}"
        return 1
    fi
}

# Validate Python packages
validate_python_packages() {
    log "INFO" "Validating Python packages..."
    
    local required_packages=(
        "torch:PyTorch"
        "transformers:HuggingFace Transformers"
        "datasets:HuggingFace Datasets"
        "numpy:NumPy"
        "pandas:Pandas"
        "matplotlib:Matplotlib"
        "scikit-learn:Scikit-learn"
    )
    
    local optional_packages=(
        "inspect-ai:Inspect AI"
        "accelerate:Accelerate"
        "evaluate:Evaluate"
        "seaborn:Seaborn"
        "plotly:Plotly"
        "typer:Typer"
        "rich:Rich"
        "pytest:Pytest"
    )
    
    # Check required packages
    for pkg in "${required_packages[@]}"; do
        local module="${pkg%%:*}"
        local name="${pkg##*:}"
        
        if python3 -c "import $module" 2>/dev/null; then
            local version=$(python3 -c "import $module; print($module.__version__)" 2>/dev/null || echo "unknown")
            log "INFO" "✅ $name: $module ($version)"
        else
            log "ERROR" "❌ $name: $module (MISSING)"
        fi
    done
    
    # Check optional packages
    for pkg in "${optional_packages[@]}"; do
        local module="${pkg%%:*}"
        local name="${pkg##*:}"
        
        if python3 -c "import $module" 2>/dev/null; then
            local version=$(python3 -c "import $module; print($module.__version__)" 2>/dev/null || echo "unknown")
            log "INFO" "✅ $name: $module ($version)"
        else
            log "WARN" "⚠️  $name: $module (OPTIONAL - NOT FOUND)"
        fi
    done
}

# Platform-specific validation
validate_platform_specific() {
    log "INFO" "Performing platform-specific validation..."
    
    case "$PLATFORM_OS" in
        "Windows")
            log "INFO" "Windows-specific checks:"
            
            # Check for WSL or Git Bash
            if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
                log "INFO" "✅ Running in Git Bash, Cygwin, or Windows"
            else
                log "WARN" "⚠️  Not running in Git Bash, Cygwin, or Windows - some features may not work"
            fi
            
            # Check for Windows-specific tools
            if command -v powershell >/dev/null 2>&1; then
                log "INFO" "✅ PowerShell available"
            else
                log "WARN" "⚠️  PowerShell not available"
            fi
            ;;
            
        "Linux")
            log "INFO" "Linux-specific checks:"
            
            # Check for system package manager
            if command -v apt >/dev/null 2>&1; then
                log "INFO" "✅ APT package manager available"
            elif command -v yum >/dev/null 2>&1; then
                log "INFO" "✅ YUM package manager available"
            elif command -v pacman >/dev/null 2>&1; then
                log "INFO" "✅ Pacman package manager available"
            else
                log "WARN" "⚠️  No common package manager found"
            fi
            
            # Check for CUDA (if available)
            if command -v nvidia-smi >/dev/null 2>&1; then
                local cuda_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n1)
                log "INFO" "✅ NVIDIA GPU detected (Driver: $cuda_version)"
            else
                log "INFO" "ℹ️  No NVIDIA GPU detected"
            fi
            ;;
            
        "macOS")
            log "INFO" "macOS-specific checks:"
            
            # Check for Homebrew
            if command -v brew >/dev/null 2>&1; then
                log "INFO" "✅ Homebrew available"
            else
                log "WARN" "⚠️  Homebrew not available"
            fi
            
            # Check for Xcode command line tools
            if command -v xcode-select >/dev/null 2>&1; then
                log "INFO" "✅ Xcode command line tools available"
            else
                log "WARN" "⚠️  Xcode command line tools not available"
            fi
            ;;
            
        *)
            log "WARN" "⚠️  Unknown platform: $PLATFORM_OS"
            ;;
    esac
}

# Environment validation
validate_environment() {
    log "INFO" "Validating environment..."
    
    # Check Python version
    if [[ -n "$PLATFORM_PYTHON" ]]; then
        local python_version=$($PLATFORM_PYTHON --version 2>&1 | cut -d' ' -f2)
        local major_version=$(echo "$python_version" | cut -d'.' -f1)
        local minor_version=$(echo "$python_version" | cut -d'.' -f2)
        
        if [ "$major_version" -eq 3 ] && [ "$minor_version" -ge 7 ]; then
            log "INFO" "✅ Python version compatible: $python_version"
        else
            log "ERROR" "❌ Python version incompatible: $python_version (requires 3.7+)"
            return 1
        fi
    fi
    
    # Check available memory
    local memory_gb=""
    case "$PLATFORM_OS" in
        "Linux")
            memory_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "unknown")
            ;;
        "macOS")
            memory_gb=$(sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}' || echo "unknown")
            ;;
        "Windows")
            memory_gb=$(wmic computersystem get TotalPhysicalMemory 2>/dev/null | awk 'NR==2{print int($1/1024/1024/1024)}' || echo "unknown")
            ;;
    esac
    
    if [[ "$memory_gb" != "unknown" ]]; then
        if [ "$memory_gb" -ge 8 ]; then
            log "INFO" "✅ Sufficient memory: ${memory_gb}GB"
        else
            log "WARN" "⚠️  Low memory: ${memory_gb}GB (recommended: 8GB+)"
        fi
    else
        log "WARN" "⚠️  Could not determine available memory"
    fi
    
    # Check disk space
    local disk_space=""
    case "$PLATFORM_OS" in
        "Linux"|"macOS")
            disk_space=$(df -h . | awk 'NR==2{print $4}' || echo "unknown")
            ;;
        "Windows")
            disk_space=$(df -h . | awk 'NR==2{print $4}' || echo "unknown")
            ;;
    esac
    
    if [[ "$disk_space" != "unknown" ]]; then
        log "INFO" "✅ Available disk space: $disk_space"
    else
        log "WARN" "⚠️  Could not determine available disk space"
    fi
}

# Main function
main() {
    log "INFO" "Starting platform detection and dependency validation"
    
    detect_platform
    validate_dependencies
    validate_environment
    
    log "INFO" "Platform detection and dependency validation completed"
    
    # Export validation results
    export DEPENDENCY_VALIDATION_PASSED="true"
    export PLATFORM_COMPATIBLE="true"
}

# Run main function
main "$@"
