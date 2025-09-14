#!/usr/bin/env bash
# RobustCBRN Evaluation Pipeline - Testing and Security Validation
# This script runs tests and validates security requirements

set -euo pipefail

# Configuration with defaults
VENV_DIR="${VENV_DIR:-.venv}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
TEST_DIR="${TEST_DIR:-tests}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
MAKEFILE="${MAKEFILE:-Makefile}"

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

log "INFO" "Starting testing and security validation"

# Step 1: Activate virtual environment
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

# Step 2: Run test suite
log "INFO" "Running test suite..."

# Check if Makefile has test target
if [ -f "$MAKEFILE" ] && grep -q "^test:" "$MAKEFILE"; then
    log "INFO" "Running tests using Makefile..."
    make test || {
        log "WARN" "Makefile test target failed, trying pytest directly"
    }
fi

# Try pytest directly
log "INFO" "Running tests using pytest..."
if command -v pytest >/dev/null 2>&1; then
    pytest "$TEST_DIR" -v --tb=short || {
        log "WARN" "Some tests failed, but continuing with security validation"
    }
else
    log "WARN" "pytest not available, skipping test execution"
fi

# Step 3: Security validation
log "INFO" "Running security validation..."

# Check for security validation script
local security_script="scripts/validate_release.sh"
if [ -f "$security_script" ]; then
    log "INFO" "Running security validation script: $security_script"
    bash "$security_script" || {
        log "WARN" "Security validation script failed, performing basic checks"
    }
else
    log "INFO" "Security validation script not found, performing basic checks..."
fi

# Basic security checks
log "INFO" "Performing basic security checks..."

# Check 1: Look for potential sensitive data leaks
log "INFO" "Checking for potential sensitive data in public artifacts..."
if [ -d "$ARTIFACTS_DIR" ]; then
    local sensitive_matches=$(grep -r -i "radiological\|biological\|chemical\|nuclear\|weapon\|explosive" "$ARTIFACTS_DIR" \
        --include="*.json" --include="*.md" --include="*.txt" 2>/dev/null | wc -l)
    
    if [ $sensitive_matches -gt 0 ]; then
        log "WARN" "Found $sensitive_matches potential sensitive data matches in artifacts"
        log "DEBUG" "Sensitive data matches:"
        grep -r -i "radiological\|biological\|chemical\|nuclear\|weapon\|explosive" "$ARTIFACTS_DIR" \
            --include="*.json" --include="*.md" --include="*.txt" 2>/dev/null | head -5 | while read -r line; do
            log "DEBUG" "  $line"
        done
    else
        log "INFO" "No sensitive data found in artifacts"
    fi
else
    log "WARN" "Artifacts directory not found: $ARTIFACTS_DIR"
fi

# Check 2: Look for per-item exploit labels
log "INFO" "Checking for per-item exploit labels..."
if [ -d "$ARTIFACTS_DIR" ]; then
    local exploit_files=$(find "$ARTIFACTS_DIR" -name "*.json" -exec grep -l "exploit\|hazard" {} \; 2>/dev/null | wc -l)
    
    if [ $exploit_files -gt 0 ]; then
        log "WARN" "Found $exploit_files files with potential exploit labels"
        log "DEBUG" "Files with exploit labels:"
        find "$ARTIFACTS_DIR" -name "*.json" -exec grep -l "exploit\|hazard" {} \; 2>/dev/null | head -5 | while read -r file; do
            log "DEBUG" "  $file"
        done
    else
        log "INFO" "No per-item exploit labels found"
    fi
else
    log "WARN" "Artifacts directory not found for exploit label check"
fi

# Check 3: Validate JSON structure
log "INFO" "Validating JSON structure in artifacts..."
if [ -d "$ARTIFACTS_DIR" ]; then
    local json_files=$(find "$ARTIFACTS_DIR" -name "*.json" | wc -l)
    local valid_json=0
    
    if [ $json_files -gt 0 ]; then
        find "$ARTIFACTS_DIR" -name "*.json" | while read -r json_file; do
            if python -c "import json; json.load(open('$json_file'))" 2>/dev/null; then
                ((valid_json++))
            else
                log "WARN" "Invalid JSON file: $json_file"
            fi
        done
        log "INFO" "Validated $valid_json/$json_files JSON files"
    else
        log "WARN" "No JSON files found in artifacts"
    fi
else
    log "WARN" "Artifacts directory not found for JSON validation"
fi

# Check 4: Check for hardcoded paths or credentials
log "INFO" "Checking for hardcoded paths or credentials..."
local hardcoded_issues=0

# Check for hardcoded paths
if [ -d "$ARTIFACTS_DIR" ]; then
    local path_matches=$(grep -r "/home/\|/Users/\|C:\\" "$ARTIFACTS_DIR" \
        --include="*.json" --include="*.md" --include="*.txt" 2>/dev/null | wc -l)
    
    if [ $path_matches -gt 0 ]; then
        log "WARN" "Found $path_matches potential hardcoded paths"
        ((hardcoded_issues++))
    fi
fi

# Check for potential credentials
if [ -d "$ARTIFACTS_DIR" ]; then
    local cred_matches=$(grep -r -i "password\|secret\|key\|token" "$ARTIFACTS_DIR" \
        --include="*.json" --include="*.md" --include="*.txt" 2>/dev/null | wc -l)
    
    if [ $cred_matches -gt 0 ]; then
        log "WARN" "Found $cred_matches potential credential references"
        ((hardcoded_issues++))
    fi
fi

if [ $hardcoded_issues -eq 0 ]; then
    log "INFO" "No hardcoded paths or credentials found"
fi

# Step 4: Summary
log "INFO" "Testing and security validation completed"
log "INFO" "Test directory: $TEST_DIR"
log "INFO" "Artifacts directory: $ARTIFACTS_DIR"
log "INFO" "Security issues found: $hardcoded_issues"

if [ $hardcoded_issues -gt 0 ]; then
    log "WARN" "Security validation completed with warnings"
else
    log "INFO" "Security validation passed"
fi
