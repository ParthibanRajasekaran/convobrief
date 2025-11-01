.PHONY: setup lint typecheck test check run dashboard clean docker help

help:
	@echo "Available commands:"
	@echo "  make setup       - Install dependencies"
	@echo "  make lint        - Run ruff and black"
	@echo "  make lint-fix    - Auto-fix linting issues"
	@echo "  make typecheck   - Run mypy"
	@echo "  make test        - Run all tests"
	@echo "  make test-unit   - Run unit tests only"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make check       - Run all checks (lint + typecheck + test)"
	@echo "  make run         - Run development server (FastAPI)"
	@echo "  make dashboard   - Run Streamlit dashboard (requires backend running)"
	@echo "  make clean       - Remove generated files"
	@echo "  make docker      - Build Docker image"
	@echo "  make docker-run  - Run Docker container"

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

dashboard:
	@echo "Starting Streamlit dashboard..."
	@echo "Make sure the backend is running on http://localhost:8000"
	@echo "Dashboard will open at http://localhost:8501"
	poetry run streamlit run dashboard.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf artifacts/*

docker:
	docker build -t insightsvc:latest .

docker-run:
	docker run -p 8000:8000 --gpus all -v $(PWD)/.env:/app/.env insightsvc:latest
