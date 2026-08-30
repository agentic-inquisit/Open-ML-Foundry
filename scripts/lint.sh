#!/bin/bash
# Run all linters for LocalML finetune
# Usage: ./scripts/lint.sh [options]
# Options: --fix, --verbose, --module <path>

set -e

FIX=false
VERBOSE=false
PATH_FILTER="sentinel tests"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix|-f) FIX=true; shift ;;
        --verbose|-v) VERBOSE=true; shift ;;
        --module|-m) PATH_FILTER="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "🔍 Running LocalML finetune Linters"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

TOTAL_ISSUES=0

# Black (code formatting)
echo "📝 Running Black (code formatter)..."
if [ "$FIX" = true ]; then
    black $PATH_FILTER --line-length=100
    echo "  ✓ Code formatted"
else
    if black --check $PATH_FILTER --line-length=100 2>/dev/null; then
        echo "  ✓ Code formatting: OK"
    else
        echo "  ⚠ Code formatting issues found"
        TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
    fi
fi

echo ""

# isort (import sorting)
echo "📚 Running isort (import sorter)..."
if [ "$FIX" = true ]; then
    isort $PATH_FILTER --profile=black --line-length=100
    echo "  ✓ Imports sorted"
else
    if isort --check-only $PATH_FILTER --profile=black --line-length=100 2>/dev/null; then
        echo "  ✓ Import sorting: OK"
    else
        echo "  ⚠ Import sorting issues found"
        TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
    fi
fi

echo ""

# Flake8 (style checking)
echo "🔧 Running Flake8 (style checker)..."
if flake8 $PATH_FILTER --max-line-length=100 --extend-ignore=E203,W503 2>/dev/null; then
    echo "  ✓ Style checking: OK"
else
    echo "  ⚠ Style issues found (see above)"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
fi

echo ""

# MyPy (type checking)
echo "🏷️  Running MyPy (type checker)..."
if mypy $PATH_FILTER --ignore-missing-imports --no-error-summary 2>/dev/null; then
    echo "  ✓ Type checking: OK"
else
    echo "  ⚠ Type issues found"
    if [ "$VERBOSE" = true ]; then
        mypy $PATH_FILTER --ignore-missing-imports
    fi
fi

echo ""

# Pylint (static analysis)
echo "🔎 Running Pylint (static analyzer)..."
if pylint $PATH_FILTER --exit-zero --max-line-length=100 2>/dev/null | grep -q "Your code"; then
    PYLINT_SCORE=$(pylint $PATH_FILTER --exit-zero --max-line-length=100 2>/dev/null | grep "Your code" | awk '{print $NF}' | cut -d'/' -f1)
    echo "  ✓ Pylint score: $PYLINT_SCORE/10"
else
    echo "  ⚠ Pylint analysis issue"
fi

echo ""

# Summary
echo "📊 Linting Summary:"
if [ $TOTAL_ISSUES -eq 0 ]; then
    echo "  ✅ All checks passed!"
else
    echo "  ⚠️  $TOTAL_ISSUES check(s) need attention"
fi

echo ""
echo "💡 Commands:"
echo "  Fix issues: ./scripts/lint.sh --fix"
echo "  Verbose output: ./scripts/lint.sh --verbose"
echo "  Specific module: ./scripts/lint.sh --module sentinel/cli"
echo "  Combined: ./scripts/lint.sh --fix --verbose"
echo ""

exit $TOTAL_ISSUES
