.PHONY: help db lint format run clean

help:
	@echo "Available commands:"
	@echo "  make db      - Open the DuckDB database"
	@echo "  make lint    - Lint the codebase using ruff"
	@echo "  make format  - Format the codebase using ruff"
	@echo "  make run     - Run the main application"
	@echo "  make clean   - Remove __pycache__ directories"

db:
	duckdb src/data/aapl.db

lint:
	uvx ruff check --fix

format:
	uvx ruff format

run:
	uv run src/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
