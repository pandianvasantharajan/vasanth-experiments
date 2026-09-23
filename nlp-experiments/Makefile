.PHONY: install notebook clean help

help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make notebook   - Launch Jupyter notebook"
	@echo "  make clean      - Clean generated files"

install:
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
	python -m spacy download en_core_web_sm

notebook:
	jupyter notebook notebooks/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf results/*
