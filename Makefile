.PHONY: help db lint format run

help:
	@echo "Available commands:"
	@echo "  make db      - Open the DuckDB database"
	@echo "  make lint    - Lint the codebase using ruff"
	@echo "  make format  - Format the codebase using ruff"
	@echo "  make run     - Run the main application"

db:
	duckdb src/data/aapl.db

lint:
	uvx ruff check --fix

format:
	uvx ruff format

run:
	uv run src/main.py
