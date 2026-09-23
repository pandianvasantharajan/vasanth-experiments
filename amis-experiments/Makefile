# AMIS Experiments - Makefile

.PHONY: help install notebook clean test lint

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies"
	@echo "  make notebook  - Launch Jupyter notebook"
	@echo "  make clean     - Clean generated files"
	@echo "  make lint      - Run code linting"

install:
	pip install -r requirements.txt

notebook:
	jupyter notebook notebooks/

clean:
	rm -rf results/*
	rm -rf data/__pycache__
	rm -rf utils/__pycache__
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

lint:
	flake8 utils/
