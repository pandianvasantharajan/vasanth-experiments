# GAN Model - Makefile

.PHONY: help install clean test notebook train

help:
	@echo "GAN Model - Available Commands"
	@echo "=============================="
	@echo "  make install    - Install dependencies"
	@echo "  make notebook   - Launch Jupyter notebook"
	@echo "  make clean      - Clean generated files"
	@echo "  make test       - Run tests"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

notebook:
	@echo "Launching Jupyter notebook..."
	jupyter notebook notebooks/image_generation_gan.ipynb

clean:
	@echo "Cleaning generated files..."
	rm -rf results/*.png
	rm -rf models/*.h5
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

test:
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "✅ Tests complete!"
