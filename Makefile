.PHONY: help run

help:
	@echo "Available commands:"
	@echo "  make run    - Run the application"

run:
	uv run src/main.py
