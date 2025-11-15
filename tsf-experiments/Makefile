# TSF Experiments - Makefile

.PHONY: help install notebook clean

help:
	@echo "TSF Experiments - Available Commands"
	@echo "===================================="
	@echo "  make install    - Install dependencies"
	@echo "  make notebook   - Launch Jupyter notebook"
	@echo "  make clean      - Clean generated files"
	@echo ""

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Installation complete!"

notebook:
	@echo "Launching Jupyter notebook..."
	jupyter notebook notebooks/ar1_model_analysis.ipynb

clean:
	@echo "Cleaning generated files..."
	rm -rf results/*.png results/*.csv results/*.txt
	rm -rf models/*.pkl
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"
