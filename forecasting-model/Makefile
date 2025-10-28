# Makefile for Air Temperature Forecasting Project

.PHONY: help install update clean format lint test notebook lab run-analysis

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies with Poetry"
	@echo "  install-dev  - Install with development dependencies"
	@echo "  update       - Update all dependencies"
	@echo "  clean        - Clean cache and temporary files"
	@echo "  format       - Format code with black and isort"
	@echo "  lint         - Run linting with flake8 and mypy"
	@echo "  test         - Run tests with pytest"
	@echo "  notebook     - Start Jupyter notebook"
	@echo "  lab          - Start JupyterLab"
	@echo "  run-analysis - Run the main analysis notebook"
	@echo "  shell        - Activate Poetry shell"

# Installation and environment setup
install:
	poetry install

install-dev:
	poetry install --with dev

install-dl:
	poetry install -E deep-learning

update:
	poetry update

# Code quality
format:
	poetry run black .
	poetry run isort .

lint:
	poetry run flake8 .
	poetry run mypy .

test:
	poetry run pytest

# Jupyter commands
notebook:
	poetry run jupyter notebook

lab:
	poetry run jupyter lab

run-analysis:
	poetry run jupyter nbconvert --execute --to notebook --inplace notebooks/air_temperature_forecasting.ipynb

# Utilities
shell:
	poetry shell

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf .coverage htmlcov/

# Environment info
info:
	@echo "Python version:"
	@poetry run python --version
	@echo "\nPoetry version:"
	@poetry --version
	@echo "\nInstalled packages:"
	@poetry show --tree