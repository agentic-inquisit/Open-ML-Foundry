# Development Scripts

Convenient shell scripts for common development tasks.

## Quick Start

```bash
# Setup development environment
./scripts/setup-dev.sh

# Run tests
./scripts/test.sh

# Run linters
./scripts/lint.sh

# Build documentation
./scripts/build-docs.sh --serve

# Run benchmarks
./scripts/benchmark.sh
```

## Scripts

### setup-dev.sh

**Setup development environment for local development.**

```bash
./scripts/setup-dev.sh
```

What it does:
- Checks Python version (3.8+ required)
- Creates virtual environment
- Installs dependencies
- Installs pre-commit hooks
- Verifies installation

**Output:**
```
✅ Development environment setup complete!

Next steps:
  1. Activate environment: source venv/bin/activate
  2. Run tests: make test
  3. Start services: make docker-up
```

### test.sh

**Run full test suite with optional coverage and filtering.**

```bash
# Run all tests
./scripts/test.sh

# Verbose output
./scripts/test.sh --verbose

# With coverage report
./scripts/test.sh --coverage

# Parallel testing (fast)
./scripts/test.sh --fast

# Specific module
./scripts/test.sh --module cli_structure

# Combine options
./scripts/test.sh --coverage --verbose --module dataset_browser
```

**Options:**
- `--verbose, -v` — Verbose output with long tracebacks
- `--coverage, -c` — Generate coverage report (HTML)
- `--fast, -f` — Run tests in parallel
- `--module, -m` — Test specific module

**Output:**
```
🧪 Running Open ML Foundry Test Suite

📋 Test Configuration:
  Module: All
  Verbose: false
  Coverage: false
  Parallel: false

🏃 Running tests...
====== test session starts ======
...
====== 30 passed in 2.15s ======

✅ All tests passed!
```

### lint.sh

**Run all linters (Black, isort, Flake8, MyPy, Pylint).**

```bash
# Check code quality
./scripts/lint.sh

# Auto-fix issues
./scripts/lint.sh --fix

# Verbose output
./scripts/lint.sh --verbose

# Check specific module
./scripts/lint.sh --module sentinel/cli

# Fix all issues automatically
./scripts/lint.sh --fix --verbose
```

**Options:**
- `--fix, -f` — Auto-fix formatting issues
- `--verbose, -v` — Verbose output
- `--module, -m` — Check specific module/path

**Linters Run:**
1. **Black** — Code formatting
2. **isort** — Import sorting
3. **Flake8** — Style checking
4. **MyPy** — Type checking
5. **Pylint** — Static analysis

**Output:**
```
🔍 Running Open ML Foundry Linters

📝 Running Black (code formatter)...
  ✓ Code formatting: OK

📚 Running isort (import sorter)...
  ✓ Import sorting: OK

🔧 Running Flake8 (style checker)...
  ✓ Style checking: OK

🏷️  Running MyPy (type checker)...
  ✓ Type checking: OK

🔎 Running Pylint (static analyzer)...
  ✓ Pylint score: 9.2/10

📊 Linting Summary:
  ✅ All checks passed!
```

### benchmark.sh

**Run performance benchmarks on models.**

```bash
# Default benchmark (ResNet50)
./scripts/benchmark.sh

# Specific model
./scripts/benchmark.sh --model yolov5s

# Custom dataset
./scripts/benchmark.sh --dataset ./my-data

# Use GPU
./scripts/benchmark.sh --gpu

# Save to file
./scripts/benchmark.sh --output results.json

# All options
./scripts/benchmark.sh --model resnet50 --dataset ./data --output bench.json --gpu
```

**Options:**
- `--model, -m` — Model name (default: resnet50)
- `--dataset, -d` — Dataset path
- `--output, -o` — Output file (default: benchmark_results.json)
- `--gpu` — Use GPU (CUDA)

**Benchmarks Measure:**
- Inference latency (ms)
- Throughput (images/sec)
- Memory usage (MB)
- Training time
- Model size

**Output:**
```
⚡ Running Open ML Foundry Benchmarks

📊 Benchmark Configuration:
  Model: resnet50
  Dataset: Using default test dataset
  Output: benchmark_results.json
  GPU: false

✓ CPU mode

🏃 Running benchmarks...
(This may take several minutes...)

✅ Benchmarks completed!

📈 Results saved to: benchmark_results.json

📊 Summary:
  Inference Latency: 45.2ms
  Throughput: 22.1 fps
  Memory: 456MB
```

### build-docs.sh

**Build documentation and optionally serve it.**

```bash
# Build documentation
./scripts/build-docs.sh

# Build and serve locally
./scripts/build-docs.sh --serve

# Clean old build and rebuild
./scripts/build-docs.sh --clean

# Build PDF (if Sphinx available)
./scripts/build-docs.sh --format pdf

# Clean and serve
./scripts/build-docs.sh --clean --serve
```

**Options:**
- `--serve, -s` — Build and serve on localhost:8000
- `--clean, -c` — Clean build artifacts before building
- `--format, -f` — Output format (html, pdf, etc)

**Output:**
```
📚 Building Open ML Foundry Documentation

📋 Documentation Configuration:
  Format: html
  Serve: false
  Clean: false

🏗️  Building documentation...
  Using Sphinx (professional docs builder)
  
✅ Documentation built!

📊 Build Results:
  HTML files: 25
  Output directory: docs/_build/html
  Main page: docs/_build/html/index.html

💡 To view documentation:
  ./scripts/build-docs.sh --serve
```

Visit `http://localhost:8000` to browse docs.

## Common Workflows

### Before Committing Code

```bash
./scripts/lint.sh --fix
./scripts/test.sh
```

### Before Publishing Release

```bash
./scripts/lint.sh
./scripts/test.sh --coverage
./scripts/benchmark.sh
./scripts/build-docs.sh
```

### During Development (Watch Mode)

```bash
# Setup once
./scripts/setup-dev.sh

# Then continuously run
./scripts/test.sh --fast  # Quick tests
./scripts/lint.sh --fix   # Auto-fix issues
```

### Setup New Developer

```bash
./scripts/setup-dev.sh
./scripts/test.sh --verbose
./scripts/build-docs.sh --serve
```

## Makefile Alternative

These scripts wrap Makefile targets. You can also use:

```bash
make test              # Like ./scripts/test.sh
make lint              # Like ./scripts/lint.sh --fix
make docs-serve        # Like ./scripts/build-docs.sh --serve
make dev               # Like ./scripts/setup-dev.sh
```

View all targets:
```bash
make help
```

## Troubleshooting

**"Permission denied" error:**
```bash
chmod +x scripts/*.sh
```

**"venv not found" after setup:**
```bash
# Reactivate virtual environment
source venv/bin/activate
```

**Scripts not finding pytest/black:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Or run setup again
./scripts/setup-dev.sh
```

**Benchmark fails:**
```bash
# Ensure model is imported
sentinel model import --model resnet50

# Run with verbose output
python benchmarks/run_benchmarks.py -v
```

## Contributing

When adding new scripts:
1. Use `#!/bin/bash` shebang
2. Add `set -e` to exit on error
3. Include help message in comments
4. Add colored output (echo "✅ Success")
5. Document in this README

## Platform Support

- ✅ Linux (Ubuntu, Debian, etc)
- ✅ macOS (Intel & Apple Silicon)
- ⚠️  Windows (WSL2 recommended)

For Windows native support, see Makefile targets.
