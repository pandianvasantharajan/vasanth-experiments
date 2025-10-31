# Makefile for YouTube Video Analyser Model

.PHONY: help install install-dev test lint format clean build docs run-example

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install the package and dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black"
	@echo "  clean        - Clean build artifacts"
	@echo "  build        - Build the package"
	@echo "  docs         - Generate documentation"
	@echo "  run-example  - Run basic example"
	@echo "  setup-spacy  - Download spaCy language model"

# Installation
install:
	poetry install

install-dev:
	poetry install --with dev

# Testing
test:
	poetry run pytest tests/ -v

test-coverage:
	poetry run pytest tests/ --cov=src --cov-report=html --cov-report=term

# Code quality
lint:
	poetry run flake8 src/ tests/
	poetry run mypy src/

format:
	poetry run black src/ tests/ examples/

format-check:
	poetry run black --check src/ tests/ examples/

# Build and clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build: clean
	poetry build

# Documentation
docs:
	@echo "Generate documentation here"

# Examples
run-example:
	cd examples && python basic_usage.py

run-advanced-example:
	cd examples && python advanced_usage.py

# Setup
setup-spacy:
	poetry run python -m spacy download en_core_web_sm

# Development helpers
dev-setup: install-dev setup-spacy
	poetry run pre-commit install

# CLI shortcuts
analyze:
	poetry run youtube-analyzer analyze $(URL)

transcript:
	poetry run youtube-analyzer transcript $(URL)

search:
	poetry run youtube-analyzer search "$(QUERY)"

stats:
	poetry run youtube-analyzer stats

# Docker (if needed)
docker-build:
	docker build -t youtube-analyzer .

docker-run:
	docker run -it --rm -v $(PWD):/app youtube-analyzer

# Quick development test
quick-test:
	poetry run python -c "from src.youtube_analyzer import YouTubeAnalyzer; print('✓ Import successful')"