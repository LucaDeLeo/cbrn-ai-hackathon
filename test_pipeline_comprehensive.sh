#!/usr/bin/env bash
# Comprehensive Pipeline Test Script
# Tests the new RobustCBRN pipeline integration

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test configuration
TEST_DIR="test_pipeline_output"
LOG_FILE="pipeline_test.log"

# Logging function
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] [$level] $message" | tee -a "$LOG_FILE"
}

# Test result tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_TOTAL=0

# Test function
run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_exit_code="${3:-0}"
    
    ((TESTS_TOTAL++))
    log "INFO" "Running test: $test_name"
    
    if eval "$test_command" >/dev/null 2>&1; then
        local exit_code=$?
        if [ $exit_code -eq $expected_exit_code ]; then
            echo -e "${GREEN}✅ PASS${NC}: $test_name"
            ((TESTS_PASSED++))
            log "PASS" "$test_name"
        else
            echo -e "${RED}❌ FAIL${NC}: $test_name (exit code: $exit_code, expected: $expected_exit_code)"
            ((TESTS_FAILED++))
            log "FAIL" "$test_name (exit code: $exit_code, expected: $expected_exit_code)"
        fi
    else
        local exit_code=$?
        if [ $exit_code -eq $expected_exit_code ]; then
            echo -e "${GREEN}✅ PASS${NC}: $test_name"
            ((TESTS_PASSED++))
            log "PASS" "$test_name"
        else
            echo -e "${RED}❌ FAIL${NC}: $test_name (exit code: $exit_code, expected: $expected_exit_code)"
            ((TESTS_FAILED++))
            log "FAIL" "$test_name (exit code: $exit_code, expected: $expected_exit_code)"
        fi
    fi
    echo
}

# Cleanup function
cleanup() {
    log "INFO" "Cleaning up test environment"
    rm -rf "$TEST_DIR" 2>/dev/null || true
}

# Setup test environment
setup_test_env() {
    log "INFO" "Setting up test environment"
    mkdir -p "$TEST_DIR"
    rm -f "$LOG_FILE"
    log "INFO" "Test environment ready"
}

# Test 1: Script Presence and Executability
test_script_presence() {
    echo -e "${BLUE}=== Testing Script Presence and Executability ===${NC}"
    
    local scripts=(
        "scripts/run_pipeline.sh"
        "scripts/validate_platform.sh"
        "scripts/setup_env.sh"
        "scripts/discover_entry_points.sh"
        "scripts/run_sample_evaluation.sh"
        "scripts/run_full_evaluation.sh"
        "scripts/aggregate_results.sh"
        "scripts/generate_figures.sh"
        "scripts/run_tests_and_security.sh"
        "scripts/generate_report.sh"
        "scripts/final_verification.sh"
        "scripts/platform_compat.sh"
    )
    
    for script in "${scripts[@]}"; do
        run_test "Script exists: $script" "[ -f '$script' ]"
        run_test "Script is executable: $script" "[ -x '$script' ]"
    done
}

# Test 2: Makefile Integration
test_makefile_integration() {
    echo -e "${BLUE}=== Testing Makefile Integration ===${NC}"
    
    run_test "Makefile exists" "[ -f 'Makefile' ]"
    run_test "Pipeline targets in Makefile" "grep -q 'pipeline:' Makefile"
    run_test "Pipeline-validate target" "grep -q 'pipeline-validate:' Makefile"
    run_test "Pipeline-setup target" "grep -q 'pipeline-setup:' Makefile"
    run_test "Pipeline-sample target" "grep -q 'pipeline-sample:' Makefile"
    run_test "Pipeline-full target" "grep -q 'pipeline-full:' Makefile"
}

# Test 3: Documentation Integration
test_documentation_integration() {
    echo -e "${BLUE}=== Testing Documentation Integration ===${NC}"
    
    run_test "README.md exists" "[ -f 'README.md' ]"
    run_test "Pipeline documentation in README" "grep -q 'pipeline' README.md"
    run_test "Integration guide exists" "[ -f 'scripts/INTEGRATION_GUIDE.md' ]"
    run_test "Pipeline README exists" "[ -f 'scripts/PIPELINE_README.md' ]"
}

# Test 4: Script Syntax Validation
test_script_syntax() {
    echo -e "${BLUE}=== Testing Script Syntax ===${NC}"
    
    local scripts=(
        "scripts/run_pipeline.sh"
        "scripts/validate_platform.sh"
        "scripts/setup_env.sh"
        "scripts/discover_entry_points.sh"
        "scripts/run_sample_evaluation.sh"
        "scripts/platform_compat.sh"
    )
    
    for script in "${scripts[@]}"; do
        run_test "Syntax check: $script" "bash -n '$script'"
    done
}

# Test 5: Pipeline Help and Usage
test_pipeline_help() {
    echo -e "${BLUE}=== Testing Pipeline Help and Usage ===${NC}"
    
    run_test "Pipeline help option" "bash scripts/run_pipeline.sh --help"
    run_test "Pipeline help shows usage" "bash scripts/run_pipeline.sh --help 2>&1 | grep -q 'Usage:'"
    run_test "Pipeline help shows options" "bash scripts/run_pipeline.sh --help 2>&1 | grep -q 'OPTIONS:'"
}

# Test 6: Platform Detection
test_platform_detection() {
    echo -e "${BLUE}=== Testing Platform Detection ===${NC}"
    
    run_test "Platform detection script runs" "bash scripts/validate_platform.sh"
    run_test "Platform compatibility script runs" "bash scripts/platform_compat.sh"
}

# Test 7: Entry Point Discovery
test_entry_point_discovery() {
    echo -e "${BLUE}=== Testing Entry Point Discovery ===${NC}"
    
    run_test "Entry point discovery script runs" "bash scripts/discover_entry_points.sh"
}

# Test 8: Configuration Validation
test_configuration_validation() {
    echo -e "${BLUE}=== Testing Configuration Validation ===${NC}"
    
    # Test with custom configuration
    export TEST_DATASET="data/sample_sanitized.jsonl"
    export TEST_LOGS_DIR="$TEST_DIR/logs"
    export TEST_RESULTS_DIR="$TEST_DIR/results"
    
    run_test "Pipeline accepts custom dataset" "bash scripts/run_pipeline.sh --dataset '$TEST_DATASET' --steps validate"
    run_test "Pipeline accepts custom logs dir" "bash scripts/run_pipeline.sh --logs '$TEST_LOGS_DIR' --steps validate"
    run_test "Pipeline accepts custom results dir" "bash scripts/run_pipeline.sh --results '$TEST_RESULTS_DIR' --steps validate"
}

# Test 9: Error Handling
test_error_handling() {
    echo -e "${BLUE}=== Testing Error Handling ===${NC}"
    
    run_test "Pipeline handles invalid steps" "bash scripts/run_pipeline.sh --steps invalid_step" 1
    run_test "Pipeline handles missing dataset" "bash scripts/run_pipeline.sh --dataset nonexistent.jsonl --steps validate" 1
}

# Test 10: Cross-Platform Compatibility
test_cross_platform() {
    echo -e "${BLUE}=== Testing Cross-Platform Compatibility ===${NC}"
    
    run_test "Platform detection works" "bash scripts/validate_platform.sh"
    run_test "Platform compatibility layer works" "bash scripts/platform_compat.sh"
}

# Test 11: Integration Test
test_integration() {
    echo -e "${BLUE}=== Testing Full Integration ===${NC}"
    
    # Test the complete pipeline flow (validation only)
    run_test "Complete pipeline validation step" "bash scripts/run_pipeline.sh --steps validate"
    
    # Test individual components
    run_test "Platform validation component" "bash scripts/validate_platform.sh"
    run_test "Entry point discovery component" "bash scripts/discover_entry_points.sh"
}

# Main test execution
main() {
    echo -e "${GREEN}🚀 Starting RobustCBRN Pipeline Integration Test${NC}"
    echo "=================================================="
    echo
    
    setup_test_env
    
    # Run all test suites
    test_script_presence
    test_makefile_integration
    test_documentation_integration
    test_script_syntax
    test_pipeline_help
    test_platform_detection
    test_entry_point_discovery
    test_configuration_validation
    test_error_handling
    test_cross_platform
    test_integration
    
    # Test summary
    echo -e "${BLUE}=== Test Summary ===${NC}"
    echo -e "Total Tests: ${TESTS_TOTAL}"
    echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
    echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
    echo -e "Success Rate: $(echo "scale=1; $TESTS_PASSED * 100 / $TESTS_TOTAL" | bc)%"
    
    if [ $TESTS_FAILED -eq 0 ]; then
        echo -e "${GREEN}🎉 All tests passed! Pipeline integration is successful.${NC}"
        log "SUCCESS" "All tests passed - Pipeline integration successful"
        exit 0
    else
        echo -e "${RED}❌ Some tests failed. Please check the logs.${NC}"
        log "FAILURE" "Some tests failed - Check integration"
        exit 1
    fi
}

# Cleanup on exit
trap cleanup EXIT

# Run main function
main "$@"
