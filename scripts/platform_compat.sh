#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Cross-Platform Compatibility Layer
# This script provides cross-platform compatibility functions

set -euo pipefail

# Platform detection
detect_platform() {
    if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        export PLATFORM="windows"
        export PLATFORM_SHELL="bash"
        export PATH_SEPARATOR=";"
        export VENV_ACTIVATE_SUFFIX="Scripts/activate"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        export PLATFORM="macos"
        export PLATFORM_SHELL="bash"
        export PATH_SEPARATOR=":"
        export VENV_ACTIVATE_SUFFIX="bin/activate"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        export PLATFORM="linux"
        export PLATFORM_SHELL="bash"
        export PATH_SEPARATOR=":"
        export VENV_ACTIVATE_SUFFIX="bin/activate"
    else
        export PLATFORM="unknown"
        export PLATFORM_SHELL="bash"
        export PATH_SEPARATOR=":"
        export VENV_ACTIVATE_SUFFIX="bin/activate"
    fi
}

# Cross-platform path functions
normalize_path() {
    local path="$1"
    case "$PLATFORM" in
        "windows")
            # Convert forward slashes to backslashes for Windows
            echo "$path" | sed 's|/|\\|g'
            ;;
        *)
            # Keep forward slashes for Unix-like systems
            echo "$path"
            ;;
    esac
}

# Cross-platform command execution
execute_command() {
    local cmd="$1"
    local args="${2:-}"
    
    case "$PLATFORM" in
        "windows")
            # Use cmd.exe for Windows-specific commands
            if [[ "$cmd" == "python" ]] || [[ "$cmd" == "python3" ]]; then
                cmd="python"
            fi
            ;;
        *)
            # Use standard commands for Unix-like systems
            ;;
    esac
    
    if [ -n "$args" ]; then
        $cmd $args
    else
        $cmd
    fi
}

# Cross-platform virtual environment activation
activate_venv() {
    local venv_dir="$1"
    local activate_script="$venv_dir/$VENV_ACTIVATE_SUFFIX"
    
    if [ -f "$activate_script" ]; then
        source "$activate_script"
        return 0
    else
        return 1
    fi
}

# Cross-platform file operations
create_directory() {
    local dir="$1"
    case "$PLATFORM" in
        "windows")
            mkdir -p "$dir" 2>/dev/null || {
                # Try with Windows-style path
                local win_dir=$(normalize_path "$dir")
                mkdir -p "$win_dir" 2>/dev/null || return 1
            }
            ;;
        *)
            mkdir -p "$dir" || return 1
            ;;
    esac
}

# Cross-platform process management
get_process_count() {
    case "$PLATFORM" in
        "windows")
            # Use wmic for Windows
            wmic process get processid | wc -l 2>/dev/null || echo "0"
            ;;
        *)
            # Use ps for Unix-like systems
            ps aux | wc -l 2>/dev/null || echo "0"
            ;;
    esac
}

# Cross-platform memory information
get_memory_info() {
    case "$PLATFORM" in
        "windows")
            # Use wmic for Windows
            wmic computersystem get TotalPhysicalMemory 2>/dev/null | awk 'NR==2{print int($1/1024/1024/1024)}' || echo "unknown"
            ;;
        "linux")
            # Use /proc/meminfo for Linux
            awk '/MemTotal/{print int($2/1024/1024)}' /proc/meminfo 2>/dev/null || echo "unknown"
            ;;
        "macOS")
            # Use sysctl for macOS
            sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1024/1024/1024)}' || echo "unknown"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Cross-platform disk space
get_disk_space() {
    case "$PLATFORM" in
        "windows")
            # Use dir for Windows
            dir . | awk '/bytes free/{print $3}' 2>/dev/null || echo "unknown"
            ;;
        *)
            # Use df for Unix-like systems
            df -h . | awk 'NR==2{print $4}' 2>/dev/null || echo "unknown"
            ;;
    esac
}

# Cross-platform network operations
test_network() {
    local url="${1:-https://www.google.com}"
    case "$PLATFORM" in
        "windows")
            # Use ping for Windows
            ping -n 1 "$url" >/dev/null 2>&1
            ;;
        *)
            # Use ping for Unix-like systems
            ping -c 1 "$url" >/dev/null 2>&1
            ;;
    esac
}

# Cross-platform package manager detection
detect_package_manager() {
    case "$PLATFORM" in
        "windows")
            if command -v choco >/dev/null 2>&1; then
                echo "chocolatey"
            elif command -v winget >/dev/null 2>&1; then
                echo "winget"
            else
                echo "none"
            fi
            ;;
        "linux")
            if command -v apt >/dev/null 2>&1; then
                echo "apt"
            elif command -v yum >/dev/null 2>&1; then
                echo "yum"
            elif command -v pacman >/dev/null 2>&1; then
                echo "pacman"
            else
                echo "none"
            fi
            ;;
        "macos")
            if command -v brew >/dev/null 2>&1; then
                echo "homebrew"
            else
                echo "none"
            fi
            ;;
        *)
            echo "none"
            ;;
    esac
}

# Cross-platform Python detection
detect_python() {
    local python_cmd=""
    
    # Try different Python commands
    for cmd in python3 python py; do
        if command -v "$cmd" >/dev/null 2>&1; then
            # Check if it's Python 3
            if $cmd --version 2>&1 | grep -q "Python 3"; then
                python_cmd="$cmd"
                break
            fi
        fi
    done
    
    echo "${python_cmd:-}"
}

# Cross-platform make detection
detect_make() {
    local make_cmd=""
    
    # Try different make commands
    for cmd in make gmake nmake; do
        if command -v "$cmd" >/dev/null 2>&1; then
            make_cmd="$cmd"
            break
        fi
    done
    
    echo "${make_cmd:-}"
}

# Cross-platform GPU detection
detect_gpu() {
    case "$PLATFORM" in
        "windows")
            # Use wmic for Windows
            if wmic path win32_VideoController get name 2>/dev/null | grep -q "NVIDIA"; then
                echo "nvidia"
            elif wmic path win32_VideoController get name 2>/dev/null | grep -q "AMD"; then
                echo "amd"
            else
                echo "unknown"
            fi
            ;;
        "linux"|"macos")
            # Use lspci for Linux/macOS
            if command -v lspci >/dev/null 2>&1; then
                if lspci | grep -q "NVIDIA"; then
                    echo "nvidia"
                elif lspci | grep -q "AMD"; then
                    echo "amd"
                else
                    echo "unknown"
                fi
            else
                echo "unknown"
            fi
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Initialize platform detection
detect_platform

# Export platform information
export PLATFORM
export PLATFORM_SHELL
export PATH_SEPARATOR
export VENV_ACTIVATE_SUFFIX

# Log platform information
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" >&2
}

log "INFO" "Cross-platform compatibility layer initialized"
log "INFO" "Platform: $PLATFORM"
log "INFO" "Shell: $PLATFORM_SHELL"
log "INFO" "Python: $(detect_python)"
log "INFO" "Make: $(detect_make)"
log "INFO" "Package Manager: $(detect_package_manager)"
log "INFO" "GPU: $(detect_gpu)"
log "INFO" "Memory: $(get_memory_info)GB"
log "INFO" "Disk Space: $(get_disk_space)"
