.PHONY: help db lint format run clean docker-build docker-run

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

docker-build:
	docker build . -t tech-challenge-four:latest

docker-run:
	docker run -d --name api -p 8000:8000 tech-challenge-four:latest

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
