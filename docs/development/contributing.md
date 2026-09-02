# Contributing Code

Guide for developers wanting to contribute to Open ML Foundry.

## Setup Development Environment

### 1. Clone Repository

```bash
git clone https://github.com/agentic-inquisit/open-ml-foundry.git
cd sentinel-finetune
```

### 2. Create Virtual Environment

```bash
python3.9 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Development Dependencies

```bash
make dev-install
# Or manually:
pip install -e ".[dev]"
pre-commit install
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

Branch naming:
- `feature/` — New features
- `fix/` — Bug fixes
- `docs/` — Documentation
- `refactor/` — Code refactoring

### 2. Make Changes

Write code following project conventions:

**Code Style:**
- Use Black for formatting (auto-run with pre-commit)
- Follow PEP 8
- Type hints on public functions
- Max line length: 100 characters

**Example:**
```python
def train_model(
    model: str,
    dataset: str,
    epochs: int = 10,
    batch_size: int = 32,
) -> Dict[str, float]:
    """Train a model on dataset.
    
    Args:
        model: Model name or path
        dataset: Dataset name
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Dictionary with final metrics (loss, accuracy, etc)
    """
    # Implementation
    pass
```

### 3. Run Pre-commit Hooks

Automatic on commit:
```bash
git commit -m "Add feature"
```

Or manually:
```bash
pre-commit run --all-files
```

### 4. Run Tests

```bash
# Run all tests
make test

# Run with coverage
make coverage

# Run specific test
pytest tests/test_cli_structure.py -v
```

### 5. Check Code Quality

```bash
# Lint checks
make lint

# Format code
make format

# All checks
make check
```

### 6. Update Documentation

If adding features, update docs:
- `docs/cli-guide.md` — New commands
- `docs/faq.md` — Clarifications
- `docs/tutorials/` — New examples
- Code docstrings — Implementation details

### 7. Commit and Push

```bash
git commit -m "descriptive message"
git push origin feature/your-feature-name
```

## Pull Request Process

### 1. Create PR

Go to GitHub and create PR with:

**Title:** Clear, concise (60 chars max)
- Good: "Add mixed precision training support"
- Bad: "stuff"

**Description:**
```markdown
## What
Brief description of changes

## Why
Motivation and context

## How
Technical approach

## Testing
How to verify the changes work

## Checklist
- [ ] Tests pass locally
- [ ] Code formatted with black
- [ ] Documentation updated
- [ ] No breaking changes
```

### 2. Respond to Feedback

Address review comments, update code, re-push to same branch.

### 3. Merge

Once approved, maintainer merges to main.

## Testing Guidelines

### Write Tests

Add tests in `tests/` for new features:

```python
# tests/test_new_feature.py
import pytest
from sentinel.cli.new_module import new_function

def test_new_function():
    result = new_function("input")
    assert result == "expected_output"

def test_new_function_error():
    with pytest.raises(ValueError):
        new_function("invalid")
```

### Coverage

Aim for 80%+ coverage:

```bash
make coverage
# View report: open htmlcov/index.html
```

### Test Organization

```
tests/
├── unit/              # Pure function tests
│   └── test_*.py
├── integration/       # Multi-component tests
│   └── test_*.py
├── e2e/              # Full workflow tests
│   └── test_*.py
├── fixtures/         # Test data
│   └── *.py
└── conftest.py       # Pytest configuration
```

## Code Review Checklist

When reviewing PRs:
- [ ] Code follows style guide
- [ ] Tests included and passing
- [ ] Documentation updated
- [ ] No breaking changes
- [ ] Performance impact considered
- [ ] Security implications reviewed

## Architecture Guidelines

### Directory Structure

```
sentinel/
├── cli/              # User-facing commands
├── core/             # Core ML functionality (future)
├── utils/            # Shared utilities (future)
└── __init__.py
```

### Component Coupling

- Minimize cross-component imports
- Public API through `__init__.py`
- Internal functions prefixed with `_`

### Error Handling

```python
# Good: Specific exceptions with context
if not os.path.exists(path):
    raise FileNotFoundError(f"Dataset not found: {path}")

# Avoid: Generic exceptions
raise Exception("Error")
```

## Documentation Standards

### Docstrings

Use Google-style docstrings:

```python
def train_model(model: str, epochs: int) -> Dict:
    """Train a machine learning model.
    
    Args:
        model: Name of model to train
        epochs: Number of training epochs
        
    Returns:
        Dictionary containing:
            - loss: Final training loss
            - accuracy: Final validation accuracy
            
    Raises:
        FileNotFoundError: If model not found
        ValueError: If epochs <= 0
    """
```

### Comments

Use comments sparingly:

```python
# Good: Explains WHY
if learning_rate < 1e-6:
    # Prevent learning rate from becoming too small and causing numerical instability
    learning_rate = 1e-6

# Bad: Explains WHAT (obvious from code)
# Set learning rate to 0.001
learning_rate = 0.001
```

## Performance Considerations

### Profiling

```bash
# Profile training
python -m cProfile -s cumulative sentinel/cli/commands.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler script.py
```

### Benchmarking

Add benchmarks for critical paths:

```bash
# In benchmarks/
python run_benchmarks.py
```

## Security Considerations

### Input Validation

```python
# Always validate user input
def import_model(path: str) -> None:
    if not isinstance(path, str):
        raise TypeError("path must be string")
    if ".." in path:  # Prevent path traversal
        raise ValueError("Invalid path")
```

### Dependency Security

```bash
# Check for vulnerabilities
safety check

# Update dependencies
pip list --outdated
```

## Release Process

### Version Numbers

Follow semantic versioning (MAJOR.MINOR.PATCH):
- `0.3.0` → first release
- `0.3.1` → bug fix
- `0.4.0` → new features
- `1.0.0` → stable release

### Release Checklist

- [ ] All tests pass
- [ ] All PRs merged
- [ ] Changelog updated
- [ ] Version bumped
- [ ] GitHub release created
- [ ] PyPI updated
- [ ] Docker images pushed
- [ ] Docs updated

See CHANGELOG.md for format.

## Getting Help

- **Architecture questions:** GitHub Discussions
- **Bug reports:** GitHub Issues
- **Feature requests:** GitHub Issues (labeled)
- **Code review:** Pull request comments
- **Setup help:** See installation.md

## Thank You!

Contributions are what make Open ML Foundry great. Thank you for helping!
