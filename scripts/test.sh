#!/usr/bin/env bash
# Test runner script for RobustCBRN Eval

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🧪 RobustCBRN Eval Test Runner"
echo "=============================="
echo ""

# Parse arguments
TEST_TYPE="${1:-all}"
VERBOSE="${2:-}"

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}❌ Error: Python $required_version or higher is required (found $python_version)${NC}"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "${VIRTUAL_ENV:-}" ]; then
    echo -e "${YELLOW}⚠️  Warning: Virtual environment not activated${NC}"
    echo "   Recommended: source .venv/bin/activate"
fi

# Function to run tests
run_tests() {
    local test_pattern=$1
    local description=$2

    echo -e "${GREEN}Running $description...${NC}"

    if [ "$VERBOSE" = "-v" ] || [ "$VERBOSE" = "--verbose" ]; then
        python -m unittest $test_pattern -v
    else
        python -m unittest $test_pattern
    fi

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $description passed${NC}"
    else
        echo -e "${RED}❌ $description failed${NC}"
        exit 1
    fi
    echo ""
}

# Main test execution
case "$TEST_TYPE" in
    all)
        echo "Running all tests..."
        run_tests "discover -s tests -p 'test_*.py'" "All tests"
        ;;

    smoke)
        echo "Running smoke tests..."
        run_tests "tests.test_smoke" "Smoke tests"
        ;;

    unit)
        echo "Running unit tests..."
        run_tests "discover -s tests -p 'test_*.py' -k 'not integration'" "Unit tests"
        ;;

    integration)
        echo "Running integration tests..."
        run_tests "tests.test_cli_integration" "Integration tests"
        ;;

    coverage)
        echo "Running tests with coverage..."
        echo -e "${GREEN}Calculating coverage...${NC}"

        coverage run -m unittest discover -s tests -p 'test_*.py'
        coverage report -m

        echo ""
        echo "Generating HTML report..."
        coverage html

        echo -e "${GREEN}✅ Coverage report generated in htmlcov/index.html${NC}"
        echo ""

        # Show coverage summary
        coverage report | grep TOTAL || true
        ;;

    specific)
        if [ -z "${3:-}" ]; then
            echo -e "${RED}Error: Please specify test module${NC}"
            echo "Usage: $0 specific [-v] <test_module>"
            echo "Example: $0 specific -v tests.test_pipeline"
            exit 1
        fi
        test_module="${3}"
        echo "Running specific test: $test_module"
        run_tests "$test_module" "Specific test: $test_module"
        ;;

    help|--help|-h)
        echo "Usage: $0 [test_type] [options]"
        echo ""
        echo "Test types:"
        echo "  all         - Run all tests (default)"
        echo "  smoke       - Run smoke tests only"
        echo "  unit        - Run unit tests only"
        echo "  integration - Run integration tests only"
        echo "  coverage    - Run tests with coverage report"
        echo "  specific    - Run specific test module"
        echo ""
        echo "Options:"
        echo "  -v, --verbose - Verbose output"
        echo ""
        echo "Examples:"
        echo "  $0              # Run all tests"
        echo "  $0 smoke        # Run smoke tests"
        echo "  $0 all -v       # Run all tests with verbose output"
        echo "  $0 coverage     # Run with coverage report"
        echo "  $0 specific -v tests.test_pipeline"
        exit 0
        ;;

    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

echo -e "${GREEN}🎉 Test run completed successfully!${NC}"