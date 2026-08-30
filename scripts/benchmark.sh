#!/bin/bash
# Run performance benchmarks for LocalML finetune
# Usage: ./scripts/benchmark.sh [options]
# Options: --model <name>, --dataset <path>, --output <file>, --gpu

set -e

MODEL="resnet50"
DATASET=""
OUTPUT="benchmark_results.json"
USE_GPU=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model|-m) MODEL="$2"; shift 2 ;;
        --dataset|-d) DATASET="$2"; shift 2 ;;
        --output|-o) OUTPUT="$2"; shift 2 ;;
        --gpu) USE_GPU=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "⚡ Running LocalML finetune Benchmarks"
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if benchmarks directory exists
if [ ! -f "benchmarks/run_benchmarks.py" ]; then
    echo "❌ Error: benchmarks/run_benchmarks.py not found"
    echo "Please ensure benchmarks directory exists with run_benchmarks.py"
    exit 1
fi

echo "📊 Benchmark Configuration:"
echo "  Model: $MODEL"
echo "  Dataset: ${DATASET:-Using default test dataset}"
echo "  Output: $OUTPUT"
echo "  GPU: $USE_GPU"
echo ""

# Build command
BENCH_CMD="python benchmarks/run_benchmarks.py --model $MODEL --output $OUTPUT"

if [ "$USE_GPU" = true ]; then
    BENCH_CMD="$BENCH_CMD --device cuda"
    echo "✓ GPU mode enabled (CUDA)"
else
    BENCH_CMD="$BENCH_CMD --device cpu"
    echo "✓ CPU mode"
fi

if [ -n "$DATASET" ]; then
    BENCH_CMD="$BENCH_CMD --dataset $DATASET"
fi

echo ""
echo "🏃 Running benchmarks..."
echo "(This may take several minutes...)"
echo ""

# Run benchmarks
if $BENCH_CMD; then
    echo ""
    echo "✅ Benchmarks completed!"
    echo ""
    echo "📈 Results saved to: $OUTPUT"
    echo ""

    # Try to display summary if jq is available
    if command -v jq &> /dev/null && [ -f "$OUTPUT" ]; then
        echo "📊 Summary:"
        jq '.summary' "$OUTPUT" 2>/dev/null || jq '.' "$OUTPUT" | head -20
    else
        echo "💡 View results: cat $OUTPUT"
    fi
else
    echo ""
    echo "❌ Benchmark failed"
    exit 1
fi

echo ""
echo "💡 Benchmark Commands:"
echo "  Compare models: ./scripts/benchmark.sh --model resnet50 --model vgg16"
echo "  With custom dataset: ./scripts/benchmark.sh --dataset ./my-data"
echo "  Save to file: ./scripts/benchmark.sh --output results.json"
echo "  Use GPU: ./scripts/benchmark.sh --gpu"
echo ""
