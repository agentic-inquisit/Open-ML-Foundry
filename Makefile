.PHONY: help install dev-install lint format test clean docker-build docker-up docs

help:
	@echo "Sentinel-Finetune Development Tasks"
	@echo "===================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install package in production mode"
	@echo "  make dev-install      Install with dev dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run linters (flake8, mypy, pylint)"
	@echo "  make format           Auto-format code (black, isort)"
	@echo "  make test             Run test suite"
	@echo "  make test-verbose     Run tests with verbose output"
	@echo "  make coverage         Run tests with coverage report"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker images"
	@echo "  make docker-up        Start Docker Compose services"
	@echo "  make docker-down      Stop Docker Compose services"
	@echo "  make docker-logs      View Docker logs"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs             Build documentation site"
	@echo "  make docs-serve       Serve docs locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  make pre-commit       Install git hooks"
	@echo "  make clean            Remove build artifacts"
	@echo "  make clean-all        Deep clean (including cache)"

# Install targets
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"
	pre-commit install

# Code quality targets
lint:
	@echo "Running flake8..."
	flake8 sentinel tests --max-line-length=100 --extend-ignore=E203,W503
	@echo "Running mypy..."
	mypy sentinel --ignore-missing-imports || true
	@echo "Running pylint..."
	pylint sentinel --exit-zero --max-line-length=100 || true

format:
	@echo "Formatting with black..."
	black sentinel tests --line-length=100
	@echo "Sorting imports with isort..."
	isort sentinel tests --profile=black --line-length=100

# Test targets
test:
	pytest tests/ -v --tb=short

test-verbose:
	pytest tests/ -vv --tb=long --capture=no

coverage:
	pytest tests/ --cov=sentinel --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

security-check:
	bandit -r sentinel
	safety check --json

pre-commit:
	pre-commit install
	@echo "Pre-commit hooks installed"

# Docker targets
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d
	@echo "Services running:"
	@echo "  API: http://localhost:8000"
	@echo "  Edge: http://localhost:8001"
	@echo "  Database: localhost:5432"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-clean:
	docker-compose down -v
	docker system prune -f

# Documentation targets
docs:
	@echo "Building documentation..."
	@echo "Docs are in docs/ directory"
	@echo "View with: make docs-serve"

docs-serve:
	@echo "Serving docs locally..."
	@echo "Install markdown viewer: pip install markdown"
	python3 -m http.server --directory docs 8888
	@echo "Open http://localhost:8888"

# Maintenance targets
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache .mypy_cache .coverage htmlcov/
	@echo "Clean complete"

clean-all: clean
	rm -rf .venv venv ENV/
	docker-compose down -v
	@echo "Deep clean complete"

# Quick checks
check: lint test
	@echo "All checks passed!"

# Development workflow
dev: dev-install pre-commit
	@echo "Development environment ready!"
	@echo "Next: make docker-up"

# One-time setup
setup: dev docker-build
	@echo "Setup complete!"
	@echo "Run 'make docker-up' to start services"

# Build distribution
build:
	pip install build
	python -m build
	@echo "Distribution files in dist/"

release: test lint
	@echo "Ready for release!"
	@echo "Next: git tag v0.X.X && git push --tags"

# Quick test run
quick-test:
	pytest tests/test_cli_structure.py -v

watch-tests:
	pytest-watch tests/ -v
