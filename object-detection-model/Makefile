# Makefile for Object Detection Model Project

.PHONY: help setup install install-dev clean test format lint run-notebook download-data demo

# Default target
help:
	@echo "Object Detection Model - Available Commands:"
	@echo ""
	@echo "  setup         - Initial project setup (install Poetry if needed)"
	@echo "  install       - Install project dependencies"
	@echo "  install-dev   - Install development dependencies"
	@echo "  download-data - Download sample test data"
	@echo "  demo          - Run quick demo detection"
	@echo "  run-notebook  - Start Jupyter notebook server"
	@echo "  test          - Run tests"
	@echo "  format        - Format code with black and isort"
	@echo "  lint          - Run linting checks"
	@echo "  clean         - Clean up temporary files"
	@echo ""
	@echo "Quick Start:"
	@echo "  make setup && make install && make demo"

# Project setup
setup:
	@echo "🚀 Setting up Object Detection project..."
	@which poetry > /dev/null || (echo "Installing Poetry..." && curl -sSL https://install.python-poetry.org | python3 -)
	@echo "✓ Poetry ready"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	poetry install
	@echo "✓ Dependencies installed"

install-dev:
	@echo "📦 Installing development dependencies..."
	poetry install --with dev
	@echo "✓ Development dependencies installed"

# Download sample data
download-data:
	@echo "📥 Downloading sample test data..."
	@mkdir -p data/sample_images
	@echo "Sample images will be downloaded when running the notebook"
	@echo "For full COCO dataset, please download from:"
	@echo "https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset"
	@echo "✓ Data directories created"

# Quick demo
demo:
	@echo "🎯 Running object detection demo..."
	poetry run python -c "\
	try:\
	    from ultralytics import YOLO;\
	    import requests;\
	    import os;\
	    os.makedirs('data/demo', exist_ok=True);\
	    print('Downloading test image...');\
	    response = requests.get('https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=800');\
	    with open('data/demo/test_image.jpg', 'wb') as f: f.write(response.content);\
	    print('Loading YOLO model...');\
	    model = YOLO('yolov8n.pt');\
	    print('Running detection...');\
	    results = model('data/demo/test_image.jpg');\
	    results[0].save('data/demo/detected_image.jpg');\
	    print('✓ Demo completed! Check data/demo/detected_image.jpg');\
	except Exception as e:\
	    print(f'Demo failed: {e}');\
	    print('Please run: make install && make run-notebook');\
	"

# Run Jupyter notebook
run-notebook:
	@echo "📊 Starting Jupyter notebook server..."
	@echo "Open: http://localhost:8888"
	@echo "Navigate to: notebooks/object_detection_analysis.ipynb"
	poetry run jupyter notebook --notebook-dir=. --ip=0.0.0.0 --port=8888 --no-browser

# Testing
test:
	@echo "🧪 Running tests..."
	poetry run pytest tests/ -v
	@echo "✓ Tests completed"

# Code formatting
format:
	@echo "🎨 Formatting code..."
	poetry run black .
	poetry run isort .
	@echo "✓ Code formatted"

# Linting
lint:
	@echo "🔍 Running linting checks..."
	poetry run flake8 . --max-line-length=88 --extend-ignore=E203,W503
	poetry run mypy . --ignore-missing-imports
	@echo "✓ Linting completed"

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/
	@echo "✓ Cleanup completed"

# Project status
status:
	@echo "📊 Project Status:"
	@echo "- Python version: $(shell python --version)"
	@echo "- Poetry version: $(shell poetry --version 2>/dev/null || echo 'Not installed')"
	@echo "- Project dependencies:"
	@poetry show --tree 2>/dev/null || echo "  Run 'make install' first"

# Full setup for new environment
bootstrap: setup install download-data
	@echo "🎉 Project bootstrap completed!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make demo' for a quick test"
	@echo "  2. Run 'make run-notebook' to start analysis"
	@echo "  3. Open notebooks/object_detection_analysis.ipynb"

# Development workflow
dev-setup: setup install-dev download-data
	@echo "👨‍💻 Development environment ready!"
	@echo "Pre-commit hooks and development tools installed"

# Check if all required tools are available
check-deps:
	@echo "🔍 Checking dependencies..."
	@python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo "✗ PyTorch not available"
	@python -c "import torchvision; print(f'✓ Torchvision {torchvision.__version__}')" || echo "✗ Torchvision not available"
	@python -c "import cv2; print(f'✓ OpenCV {cv2.__version__}')" || echo "✗ OpenCV not available"
	@python -c "from ultralytics import YOLO; print('✓ Ultralytics YOLO available')" || echo "✗ Ultralytics not available"
	@python -c "import matplotlib; print(f'✓ Matplotlib {matplotlib.__version__}')" || echo "✗ Matplotlib not available"

# Performance benchmark
benchmark:
	@echo "⚡ Running performance benchmark..."
	poetry run python -c "\
	import time;\
	from ultralytics import YOLO;\
	print('Loading model...');\
	model = YOLO('yolov8n.pt');\
	print('Running benchmark...');\
	start = time.time();\
	for i in range(10):\
	    model('https://ultralytics.com/images/bus.jpg', verbose=False);\
	end = time.time();\
	avg_time = (end - start) / 10;\
	fps = 1 / avg_time;\
	print(f'Average inference time: {avg_time:.3f}s');\
	print(f'Throughput: {fps:.1f} FPS');\
	"

# Documentation
docs:
	@echo "📚 Generating documentation..."
	@echo "Documentation will be available in the README.md"
	@echo "For detailed usage, see notebooks/object_detection_analysis.ipynb"

# Docker support (future enhancement)
docker-build:
	@echo "🐳 Docker support coming soon..."
	@echo "For now, use Poetry for dependency management"

# Release preparation
release-check: test lint
	@echo "✅ Release checks passed"
	@echo "Project ready for release"