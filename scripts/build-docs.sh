#!/bin/bash
# Build documentation for LocalML finetune
# Usage: ./scripts/build-docs.sh [options]
# Options: --serve, --clean, --format <fmt>

set -e

SERVE=false
CLEAN=false
FORMAT="html"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --serve|-s) SERVE=true; shift ;;
        --clean|-c) CLEAN=false; shift ;;
        --format|-f) FORMAT="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "📚 Building LocalML finetune Documentation"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if docs directory exists
if [ ! -d "docs" ]; then
    echo "❌ Error: docs/ directory not found"
    exit 1
fi

echo "📋 Documentation Configuration:"
echo "  Format: $FORMAT"
echo "  Serve: $SERVE"
echo "  Clean: $CLEAN"
echo ""

# Clean build artifacts if requested
if [ "$CLEAN" = true ]; then
    echo "🧹 Cleaning build artifacts..."
    rm -rf docs/_build/
    rm -rf build/ dist/
    echo "  ✓ Build artifacts cleaned"
    echo ""
fi

# Build documentation
echo "🏗️  Building documentation..."

# Check if Sphinx is available
if command -v sphinx-build &> /dev/null; then
    echo "  Using Sphinx (professional docs builder)"
    sphinx-build -b $FORMAT docs/ docs/_build/$FORMAT
    BUILD_DIR="docs/_build/$FORMAT"
elif command -v mkdocs &> /dev/null; then
    echo "  Using MkDocs"
    mkdocs build
    BUILD_DIR="site"
else
    echo "  Using simple Markdown viewer"
    # Create a simple index
    cd docs
    python3 -m http.server 8888 &
    HTTP_PID=$!
    echo "  ✓ Serving docs on http://localhost:8888"
    sleep 2
    BUILD_DIR="docs"
fi

echo "✅ Documentation built!"
echo ""

# Display results
if [ -d "$BUILD_DIR" ]; then
    DOC_COUNT=$(find "$BUILD_DIR" -name "*.html" 2>/dev/null | wc -l)
    if [ $DOC_COUNT -gt 0 ]; then
        echo "📊 Build Results:"
        echo "  HTML files: $DOC_COUNT"
        echo "  Output directory: $BUILD_DIR"
        echo "  Main page: $BUILD_DIR/index.html"
        echo ""
    fi
fi

# Serve documentation if requested
if [ "$SERVE" = true ]; then
    echo "🌐 Serving documentation..."

    if [ -d "$BUILD_DIR" ]; then
        cd "$BUILD_DIR"
        echo "  📖 Open http://localhost:8000 in your browser"
        echo "  Press Ctrl+C to stop"
        echo ""
        python3 -m http.server 8000
    else
        echo "❌ Build directory not found: $BUILD_DIR"
        exit 1
    fi
fi

echo ""
echo "💡 Documentation Commands:"
echo "  Build docs: ./scripts/build-docs.sh"
echo "  Build & serve: ./scripts/build-docs.sh --serve"
echo "  Clean & rebuild: ./scripts/build-docs.sh --clean"
echo "  Build PDF: ./scripts/build-docs.sh --format pdf"
echo "  View locally: python3 -m http.server 8000 --directory docs"
echo ""

echo "📖 Documentation Structure:"
echo "  Main docs: docs/index.md"
echo "  Getting Started: docs/getting-started.md"
echo "  CLI Guide: docs/cli-guide.md"
echo "  Tutorials: docs/tutorials/"
echo "  Components: docs/components/"
echo ""

if [ "$SERVE" = false ]; then
    echo "💡 To view documentation:"
    echo "  ./scripts/build-docs.sh --serve"
    echo ""
fi
