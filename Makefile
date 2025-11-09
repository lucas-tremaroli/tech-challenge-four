.PHONY: help lint format run

help:
	@echo "Available commands:"
	@echo "  make lint    - Lint the codebase using ruff"
	@echo "  make format  - Format the codebase using ruff"
	@echo "  make run     - Run the main application"

lint:
	uvx ruff check --fix

format:
	uvx ruff format

run:
	uv run src/main.py
