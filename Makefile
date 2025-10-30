.PHONY: setup lint typecheck test check run bench clean docker help

help:
	@echo "Available commands:"
	@echo "  make setup      - Install dependencies"
	@echo "  make lint       - Run ruff and black"
	@echo "  make typecheck  - Run mypy"
	@echo "  make test       - Run pytest"
	@echo "  make check      - Run all checks (lint + typecheck + test)"
	@echo "  make run        - Run development server"
	@echo "  make bench      - Run benchmarks"
	@echo "  make clean      - Remove generated files"
	@echo "  make docker     - Build Docker image"

setup:
	poetry install

lint:
	poetry run ruff check src tests
	poetry run black --check src tests

lint-fix:
	poetry run ruff check --fix src tests
	poetry run black src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest tests -v

test-unit:
	poetry run pytest tests/unit -v

test-integration:
	poetry run pytest tests/integration -v

test-cov:
	poetry run pytest --cov=src/insightsvc --cov-report=term-missing --cov-report=html

check: lint typecheck test

run:
	poetry run uvicorn insightsvc.api.app:create_app --factory --reload --host 0.0.0.0 --port 8000

bench:
	poetry run python scripts/benchmark.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf artifacts/*

docker:
	docker build -t insightsvc:latest .

docker-run:
	docker run -p 8000:8000 --gpus all -v $(PWD)/.env:/app/.env insightsvc:latest
