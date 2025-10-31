# YouTube Video Analyser Model

A robust RAG (Retrieval-Augmented Generation) pipeline for summarizing YouTube videos using transcript analysis and semantic chunking.

## Features

- **Robust transcript fetching** using youtube-transcript-api with fallback mechanisms
- **Configurable semantic chunking** with LangChain splitters or spaCy
- **Embeddings generation** using Hugging Face sentence-transformers
- **Local vector storage** with ChromaDB for efficient retrieval
- **Multiple LLM options**:
  - Local HuggingFace models (recommended for GPU users)
  - Groq API integration
  - OpenRouter/OpenAI compatibility
- **Comprehensive summarization** with optional chunk-level context

## Installation

### Using Poetry (Recommended)

```bash
# Clone or navigate to the project directory
cd youtube-video-analyser-model

# Install dependencies
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
pip install -r requirements.txt
```

### Additional Setup

For spaCy language model (optional):
```bash
python -m spacy download en_core_web_sm
```

## Configuration

Create a `.env` file in the project root:

```env
# LLM Configuration (choose one)
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_PROVIDER=groq  # Options: groq, openai, openrouter, local
LOCAL_MODEL_NAME=microsoft/DialoGPT-medium

# Chunking Configuration
CHUNKING_METHOD=langchain  # Options: langchain, spacy
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
```

## Quick Start

### Command Line Interface

```bash
# Analyze a YouTube video
youtube-analyzer "https://www.youtube.com/watch?v=VIDEO_ID"

# With custom configuration
youtube-analyzer "https://www.youtube.com/watch?v=VIDEO_ID" --chunking-method spacy --chunk-size 800
```

### Python API

```python
from youtube_analyzer import YouTubeAnalyzer

# Initialize the analyzer
analyzer = YouTubeAnalyzer()

# Analyze a video
result = analyzer.analyze_video("https://www.youtube.com/watch?v=VIDEO_ID")

print("Summary:", result.summary)
print("Key points:", result.key_points)
print("Chunks used:", len(result.context_chunks))
```

### Advanced Usage

```python
from youtube_analyzer import YouTubeAnalyzer, AnalyzerConfig

# Custom configuration
config = AnalyzerConfig(
    chunking_method="spacy",
    chunk_size=800,
    llm_provider="groq",
    embedding_model="all-mpnet-base-v2"
)

analyzer = YouTubeAnalyzer(config)

# Analyze with detailed context
result = analyzer.analyze_video(
    "https://www.youtube.com/watch?v=VIDEO_ID",
    include_context=True,
    max_chunks=10
)
```

## Project Structure

```
youtube-video-analyser-model/
├── src/youtube_analyzer/          # Main package
│   ├── __init__.py
│   ├── core/                      # Core functionality
│   │   ├── __init__.py
│   │   ├── transcript_fetcher.py  # YouTube transcript handling
│   │   ├── chunker.py             # Text chunking strategies
│   │   ├── embedder.py            # Embedding generation
│   │   ├── retriever.py           # Vector retrieval
│   │   └── summarizer.py          # LLM-based summarization
│   ├── models/                    # Data models
│   │   ├── __init__.py
│   │   └── schemas.py             # Pydantic models
│   ├── config/                    # Configuration
│   │   ├── __init__.py
│   │   └── settings.py            # Settings management
│   ├── utils/                     # Utilities
│   │   ├── __init__.py
│   │   └── helpers.py             # Helper functions
│   ├── analyzer.py                # Main analyzer class
│   └── cli.py                     # Command line interface
├── tests/                         # Test suite
├── examples/                      # Usage examples
├── config/                        # Configuration files
├── data/                          # Data storage
└── docs/                          # Documentation
```

## API Reference

### YouTubeAnalyzer

Main class for video analysis.

```python
class YouTubeAnalyzer:
    def __init__(self, config: Optional[AnalyzerConfig] = None)
    def analyze_video(self, url: str, **kwargs) -> AnalysisResult
    def get_transcript(self, url: str) -> str
    def create_embeddings(self, chunks: List[str]) -> None
    def summarize(self, query: str, context: str) -> str
```

### AnalysisResult

Result object containing analysis output.

```python
class AnalysisResult:
    summary: str
    key_points: List[str]
    context_chunks: Optional[List[str]]
    metadata: Dict[str, Any]
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Development Setup

```bash
# Install development dependencies
poetry install --with dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/

# Type checking
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

1. **Transcript not available**: Some videos may not have transcripts available
2. **API rate limits**: Consider implementing rate limiting for API calls
3. **Memory usage**: Large videos may require chunking adjustments
4. **GPU memory**: Adjust batch sizes for local model inference

### Performance Tips

- Use GPU acceleration for local models when available
- Adjust chunk sizes based on your use case
- Consider using lighter embedding models for faster processing
- Cache embeddings for repeated analysis

## Roadmap

- [ ] Support for multiple languages
- [ ] Audio extraction for videos without transcripts
- [ ] Batch processing capabilities
- [ ] Web interface
- [ ] Integration with popular video platforms
- [ ] Advanced summarization techniques