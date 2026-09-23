# NLP Experiments

Natural Language Processing experiments and analysis notebooks.

## Project Overview

This project contains various NLP experiments including:
- Semantic similarity analysis
- Text comparison and matching
- Sentence embeddings
- Contextual analysis

## Features

### Semantic Sentence Comparison
- Compare two sentences semantically
- Calculate similarity scores using multiple methods
- Analyze matching factors and features
- Visualize similarity metrics

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Download required models
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Usage

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/

# Or use the Makefile
make notebook
```

## Requirements

- Python 3.8+
- transformers
- sentence-transformers
- nltk
- scikit-learn
- spacy

## Methods Used

1. **Cosine Similarity**: Vector-based semantic similarity
2. **Word Embeddings**: Word2Vec, GloVe
3. **Transformer Models**: BERT, Sentence-BERT
4. **Traditional NLP**: TF-IDF, Jaccard similarity
5. **Linguistic Features**: POS tags, dependency parsing

## Directory Structure

```
nlp-experiments/
├── notebooks/          # Jupyter notebooks
├── results/           # Output and analysis
├── utils/             # Helper functions
├── README.md
└── requirements.txt
```

## License

MIT License
