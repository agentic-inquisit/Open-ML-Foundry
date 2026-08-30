#!/bin/bash
# Setup development environment for LocalML finetune
# Usage: ./scripts/setup-dev.sh

set -e  # Exit on error

echo "🚀 Setting up LocalML finetune development environment..."
echo ""

# Check Python version
echo "✓ Checking Python version..."
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "  Python version: $PYTHON_VERSION"

REQUIRED_MAJOR=3
REQUIRED_MINOR=8
VERSION_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
VERSION_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$VERSION_MAJOR" -lt "$REQUIRED_MAJOR" ] || \
   ([ "$VERSION_MAJOR" -eq "$REQUIRED_MAJOR" ] && [ "$VERSION_MINOR" -lt "$REQUIRED_MINOR" ]); then
    echo "❌ Error: Python 3.8+ required (found $PYTHON_VERSION)"
    exit 1
fi

# Create virtual environment
echo ""
echo "✓ Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  Virtual environment created: ./venv"
else
    echo "  Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "✓ Activating virtual environment..."
source venv/bin/activate
echo "  Virtual environment activated"

# Upgrade pip
echo ""
echo "✓ Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "  pip upgraded"

# Install dependencies
echo ""
echo "✓ Installing development dependencies..."
pip install -e ".[dev]" > /dev/null 2>&1
echo "  Dependencies installed"

# Install pre-commit hooks
echo ""
echo "✓ Installing pre-commit hooks..."
pre-commit install > /dev/null 2>&1
echo "  Pre-commit hooks installed"

# Verify installation
echo ""
echo "✓ Verifying installation..."
python -c "import sentinel; print('  sentinel module: OK')" || echo "  sentinel module: FAILED"
python -c "import pytest; print('  pytest: OK')" || echo "  pytest: FAILED"
python -c "import black; print('  black: OK')" || echo "  black: FAILED"

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📝 Next steps:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Run tests: make test"
echo "  3. Start services: make docker-up"
echo "  4. View docs: make docs-serve"
echo ""
echo "💡 Useful commands:"
echo "  make test          - Run all tests"
echo "  make lint          - Check code quality"
echo "  make format        - Auto-format code"
echo "  make docker-up     - Start services"
echo "  make help          - Show all available commands"
echo ""
