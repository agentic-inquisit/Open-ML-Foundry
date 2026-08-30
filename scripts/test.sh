#!/bin/bash
# Run full test suite for LocalML finetune
# Usage: ./scripts/test.sh [options]
# Options: --verbose, --coverage, --fast, --module <name>

set -e

VERBOSE=false
COVERAGE=false
FAST=false
MODULE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v) VERBOSE=true; shift ;;
        --coverage|-c) COVERAGE=true; shift ;;
        --fast|-f) FAST=true; shift ;;
        --module|-m) MODULE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "🧪 Running LocalML finetune Test Suite"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Determine test command
TEST_ARGS="tests/"
if [ -n "$MODULE" ]; then
    TEST_ARGS="tests/test_${MODULE}.py"
fi

if [ "$VERBOSE" = true ]; then
    TEST_ARGS="$TEST_ARGS -vv --tb=long"
else
    TEST_ARGS="$TEST_ARGS -v --tb=short"
fi

if [ "$COVERAGE" = true ]; then
    TEST_ARGS="$TEST_ARGS --cov=sentinel --cov-report=html --cov-report=term-missing"
fi

if [ "$FAST" = true ]; then
    TEST_ARGS="$TEST_ARGS -n auto"  # Use pytest-xdist for parallel testing
fi

# Run tests
echo "📋 Test Configuration:"
if [ -n "$MODULE" ]; then
    echo "  Module: $MODULE"
else
    echo "  Module: All"
fi
echo "  Verbose: $VERBOSE"
echo "  Coverage: $COVERAGE"
echo "  Parallel: $FAST"
echo ""

echo "🏃 Running tests..."
pytest $TEST_ARGS

# Report results
echo ""
echo "✅ All tests passed!"
echo ""

if [ "$COVERAGE" = true ]; then
    echo "📊 Coverage Report:"
    echo "  HTML report: htmlcov/index.html"
    echo ""
fi

echo "💡 Commands:"
echo "  Run specific module: ./scripts/test.sh --module cli_structure"
echo "  With coverage: ./scripts/test.sh --coverage"
echo "  Verbose output: ./scripts/test.sh --verbose"
echo "  Parallel tests: ./scripts/test.sh --fast"
echo ""
