.PHONY: help db lint format up down docker-build clean

help:
	@echo "Available commands:"
	@echo "  make db            - Open the DuckDB database"
	@echo "  make lint          - Lint the codebase using ruff"
	@echo "  make format        - Format the codebase using ruff"
	@echo "  make up            - Start the Docker containers"
	@echo "  make down          - Stop the Docker containers"
	@echo "  make docker-build  - Build the Docker image for the API"
	@echo "  make clean         - Remove __pycache__ directories"

db:
	duckdb src/data/aapl.db

lint:
	uvx ruff check --fix

format:
	uvx ruff format

up:
	docker compose up -d

down:
	docker compose down

docker-build:
	docker build -t tech-challenge-four-api .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
