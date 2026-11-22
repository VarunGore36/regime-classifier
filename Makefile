.PHONY: help install test lint format type-check clean build run-backtest run-walkforward run-bootstrap

help:
	@echo "Regime Classifier - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  install        Install package in development mode"
	@echo "  install-dev    Install with dev dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run ruff linter"
	@echo "  format         Format code with black/ruff"
	@echo "  type-check     Run mypy type checker"
	@echo "  pre-commit     Run pre-commit hooks"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests"
	@echo "  test-cov       Run tests with coverage"
	@echo "  test-fast      Run tests without coverage"
	@echo ""
	@echo "Backtesting:"
	@echo "  run-backtest   Run backtest with synthetic data"
	@echo "  run-walkforward Run walk-forward backtest"
	@echo "  run-bootstrap  Run bootstrap validation"
	@echo ""
	@echo "Building:"
	@echo "  build          Build package"
	@echo "  clean          Clean build artifacts"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .
	black .

type-check:
	mypy regime_classifier --ignore-missing-imports

pre-commit:
	pre-commit run --all-files

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=regime_classifier --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v --no-cov -x

run-backtest:
	python main.py backtest --synthetic --output reports/backtest.pdf

run-walkforward:
	python main.py walkforward --synthetic --output reports/walkforward.pdf

run-bootstrap:
	python main.py bootstrap --synthetic --n-paths 500

build:
	python -m build

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .mypy_cache/ .ruff_cache/ htmlcov/ __pycache__/ */__pycache__/ */*/__pycache__/
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete