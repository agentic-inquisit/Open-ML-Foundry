# Testing Infrastructure

Comprehensive testing suite for Open ML Foundry.

## Test Structure

```
tests/
├── __init__.py                 # Test package marker
├── conftest.py                 # Pytest configuration & fixtures
├── unit/                       # Unit tests (isolated component tests)
│   ├── __init__.py
│   ├── test_cli_structure.py
│   ├── test_dataset_browser.py
│   └── test_job_tracker.py
├── integration/                # Integration tests (component interaction)
│   ├── __init__.py
│   └── test_api_endpoints.py
├── e2e/                        # End-to-end tests (full workflows)
│   ├── __init__.py
│   └── test_training_workflow.py
├── fixtures/                   # Test data & fixtures
│   ├── __init__.py
│   ├── sample_data/
│   ├── sample_models/
│   └── sample_configs/
└── README.md                   # This file
```

## Running Tests

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Category

```bash
# Unit tests only
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v
```

### Run With Coverage

```bash
pytest tests/ --cov=sentinel --cov-report=html
```

### Run Specific Test File

```bash
pytest tests/unit/test_cli_structure.py -v
```

### Run Tests Matching Pattern

```bash
pytest tests/ -k "dataset" -v
```

### Run With Different Verbosity

```bash
# Verbose output
pytest tests/ -vv

# Simple output
pytest tests/ -q

# Show print statements
pytest tests/ -s
```

### Parallel Testing

```bash
# Install pytest-xdist first
pip install pytest-xdist

# Run tests in parallel
pytest tests/ -n auto
```

## Test Categories

### Unit Tests (`unit/`)

**Purpose:** Test individual components in isolation

**What to test:**
- CLI command parsing
- Dataset detection logic
- Job state transitions
- Metric calculations
- Error handling

**Example:**
```python
def test_dataset_browser_detects_classes():
    """Test dataset class detection."""
    browser = DatasetBrowser("data/test")
    classes = browser.detect_classes()
    assert len(classes) == 3
    assert "class_1" in classes
```

**Run:**
```bash
pytest tests/unit/ -v
```

### Integration Tests (`integration/`)

**Purpose:** Test component interactions

**What to test:**
- API endpoints with database
- CLI commands with file system
- Model training with data loading
- API request/response flows

**Example:**
```python
def test_train_api_endpoint(client):
    """Test training endpoint."""
    response = client.post("/train/start", json={
        "model": "resnet50",
        "dataset": "test"
    })
    assert response.status_code == 200
    assert "job_id" in response.json()
```

**Run:**
```bash
pytest tests/integration/ -v
```

### End-to-End Tests (`e2e/`)

**Purpose:** Test complete workflows

**What to test:**
- Full training pipeline (import → prepare → train → export)
- User interactions (CLI → API → DB → File system)
- Real model operations

**Example:**
```python
def test_full_training_workflow():
    """Test complete training workflow."""
    # Import model
    result = run_cli("model import --model resnet50")
    assert result.returncode == 0
    
    # Prepare dataset
    result = run_cli("dataset prepare --path ./data")
    assert result.returncode == 0
    
    # Train model
    result = run_cli("train start --model resnet50 --dataset data")
    assert result.returncode == 0
```

**Run:**
```bash
pytest tests/e2e/ -v
```

## Test Fixtures

### Location: `fixtures/`

Test data and fixtures for reproducible testing.

**Structure:**
```
fixtures/
├── sample_data/           # Test datasets
│   ├── images/
│   ├── texts/
│   └── models/
├── sample_models/         # Minimal test models
│   ├── resnet_mini.pth
│   └── config.json
└── sample_configs/        # Test configurations
    ├── training_config.yaml
    └── dataset_config.yaml
```

### Using Fixtures in Tests

```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_data_path():
    """Provide path to sample data."""
    return Path(__file__).parent / "fixtures" / "sample_data"

def test_with_fixture(sample_data_path):
    """Test using fixture."""
    assert (sample_data_path / "images").exists()
```

## Pytest Configuration

**File:** `conftest.py`

Defines:
- Pytest plugins
- Shared fixtures
- Configuration options
- Test markers

**Example:**
```python
import pytest

@pytest.fixture
def sample_model():
    """Provide sample model."""
    # Setup
    model = load_model("resnet50")
    
    # Test runs with this model
    yield model
    
    # Cleanup
    model.cleanup()
```

## Test Markers

Use markers to categorize and filter tests:

```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.slow
def test_slow_operation():
    pass

@pytest.mark.skip(reason="Not implemented yet")
def test_future_feature():
    pass
```

**Run marked tests:**
```bash
pytest tests/ -m unit           # Unit tests only
pytest tests/ -m integration    # Integration only
pytest tests/ -m "not slow"     # Exclude slow tests
```

## Coverage

### Generate Coverage Report

```bash
pytest tests/ --cov=sentinel --cov-report=html
```

Opens `htmlcov/index.html` in browser.

### Set Coverage Threshold

```bash
pytest tests/ --cov=sentinel --cov-fail-under=80
```

Fails if coverage < 80%.

### View Coverage in Terminal

```bash
pytest tests/ --cov=sentinel --cov-report=term-missing
```

Shows which lines aren't covered.

## Continuous Integration

Tests run automatically on:
- Every commit (pre-commit hooks)
- Every push (GitHub Actions)
- Every pull request (CI/CD)

**CI Configuration:** `.github/workflows/tests.yml`

## Best Practices

### 1. Test Isolation

Each test should be independent:

```python
# Good: Each test is standalone
def test_case_1():
    result = function_a()
    assert result == expected

def test_case_2():
    result = function_b()
    assert result == expected

# Bad: Test depends on another test
def test_setup():
    global_state = initialize()

def test_uses_global_state():
    assert global_state is not None  # Fails if test_setup didn't run
```

### 2. Clear Names

Use descriptive test names:

```python
# Good
def test_dataset_browser_detects_image_classification():
    pass

# Bad
def test_1():
    pass
```

### 3. Arrange-Act-Assert

Structure tests clearly:

```python
def test_model_training():
    # Arrange: Setup test data
    model = ModelTrainer()
    data = load_test_data()
    
    # Act: Perform action
    result = model.train(data)
    
    # Assert: Check results
    assert result.accuracy > 0.8
```

### 4. Test One Thing

Each test should verify one behavior:

```python
# Good
def test_model_trains_successfully():
    model.train(data)
    assert model.is_trained

def test_model_saves_to_file():
    model.save("path")
    assert Path("path").exists()

# Bad: Testing multiple things
def test_model_training_and_saving():
    model.train(data)
    model.save("path")
    assert model.is_trained
    assert Path("path").exists()
```

### 5. Use Assertions Wisely

```python
# Good
assert result == expected
assert value > 0
assert "error" not in output

# Bad
assert result  # Not specific
assert not result  # Unclear what failed
```

## Debugging Tests

### Run With Debug Output

```bash
pytest tests/ -s  # Show print statements
pytest tests/ -vv # Very verbose output
```

### Drop Into Debugger

```python
def test_something():
    result = complex_operation()
    import pdb; pdb.set_trace()  # Stops here
    assert result == expected
```

### Run Single Test

```bash
pytest tests/unit/test_cli_structure.py::test_specific_case -v
```

## CI/CD Integration

Tests run on:

```yaml
# .github/workflows/tests.yml
- Python 3.8, 3.9, 3.10, 3.11
- Ubuntu, macOS, Windows
- Coverage reporting
- Linting checks
```

**Status:** All tests must pass before merge.

## Adding New Tests

### 1. Create Test File

```bash
# Unit test
touch tests/unit/test_new_feature.py

# Integration test
touch tests/integration/test_new_feature.py
```

### 2. Write Test

```python
import pytest
from sentinel.new_module import new_function

def test_new_function():
    """Test new function."""
    result = new_function(input_data)
    assert result == expected_output

def test_new_function_with_error():
    """Test error handling."""
    with pytest.raises(ValueError):
        new_function(invalid_input)
```

### 3. Run Test

```bash
pytest tests/unit/test_new_feature.py -v
```

### 4. Check Coverage

```bash
pytest tests/ --cov=sentinel --cov-report=term-missing
```

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Testing Best Practices](https://docs.pytest.org/en/latest/goodpractices.html)
- [Fixtures Guide](https://docs.pytest.org/en/latest/fixture.html)

## Test Statistics

**Current Coverage:**
- Unit tests: 15+ tests
- Integration tests: 8+ tests
- E2E tests: 5+ tests
- Total: **30+ test cases**

**Coverage Target:** 80%+

**CI/CD Platforms:** 3 (Ubuntu, macOS, Windows)

**Python Versions:** 4 (3.8, 3.9, 3.10, 3.11)

---

**Keep tests updated as code changes!** 🧪
