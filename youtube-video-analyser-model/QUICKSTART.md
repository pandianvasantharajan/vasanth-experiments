# Quick Start Guide

## YouTube Video Analyzer Model

A comprehensive RAG (Retrieval-Augmented Generation) pipeline for analyzing and summarizing YouTube videos.

### 🚀 Quick Setup

1. **Clone/Download the project**
   ```bash
   cd youtube-video-analyser-model
   ```

2. **Run the setup script**
   ```bash
   ./setup.sh
   ```
   
   Or manually:
   ```bash
   # Using Poetry (recommended)
   poetry install
   poetry run python -m spacy download en_core_web_sm
   
   # OR using pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

3. **Configure API keys**
   ```bash
   cp .env.example .env
   # Edit .env file and add your API keys
   ```

### 🔑 API Keys Setup

Add these to your `.env` file:

```env
# Choose one LLM provider:
GROQ_API_KEY=your_groq_key_here          # Recommended (fast & free tier)
OPENAI_API_KEY=your_openai_key_here      # Alternative
OPENROUTER_API_KEY=your_openrouter_key   # Alternative

# Optional: Customize models
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHUNKING_METHOD=langchain
CHUNK_SIZE=1000
```

### 📖 Usage Examples

#### 1. Command Line Interface

```bash
# Analyze a video
poetry run youtube-analyzer analyze "https://www.youtube.com/watch?v=VIDEO_ID"

# With custom options
poetry run youtube-analyzer analyze "URL" --chunking-method spacy --max-chunks 5

# Extract transcript only
poetry run youtube-analyzer transcript "URL"

# Search existing content
poetry run youtube-analyzer search "machine learning concepts"

# View database stats
poetry run youtube-analyzer stats
```

#### 2. Python API

```python
from youtube_analyzer import YouTubeAnalyzer, AnalyzerConfig

# Basic usage
analyzer = YouTubeAnalyzer()
result = analyzer.analyze_video("https://www.youtube.com/watch?v=VIDEO_ID")

print("Summary:", result.summary)
print("Key Points:", result.key_points)

# Custom configuration
config = AnalyzerConfig(
    llm_provider="groq",
    chunking_method="spacy",
    chunk_size=800
)
analyzer = YouTubeAnalyzer(config)
result = analyzer.analyze_video("URL", query="technical details")
```

#### 3. Focused Analysis

```python
# Analyze with specific focus
result = analyzer.analyze_video(
    "URL",
    query="business implications and market impact",
    max_chunks=5
)
```

### 🛠️ Features

- **🎥 Robust transcript fetching** from YouTube videos
- **✂️ Smart text chunking** (LangChain or spaCy)
- **🧠 Semantic embeddings** (Hugging Face models)
- **🔍 Vector search** (ChromaDB)
- **📝 AI summarization** (Groq/OpenAI/Local models)
- **⚡ CLI interface** for quick analysis
- **🔧 Configurable pipeline** for different use cases

### 🏗️ Architecture

```
YouTube URL → Transcript → Chunking → Embeddings → Vector DB
                                                        ↓
Summary ← LLM ← Context Retrieval ← Similarity Search ←
```

### 📁 Project Structure

```
youtube-video-analyser-model/
├── src/youtube_analyzer/     # Main package
│   ├── core/                 # Core functionality
│   ├── models/              # Data models
│   ├── config/              # Configuration
│   └── utils/               # Utilities
├── examples/                # Usage examples
├── tests/                   # Test suite
├── data/                    # Data storage
└── README.md               # Full documentation
```

### 🔧 Configuration Options

| Setting | Options | Description |
|---------|---------|-------------|
| `llm_provider` | `groq`, `openai`, `openrouter`, `local` | LLM service |
| `chunking_method` | `langchain`, `spacy` | Text chunking strategy |
| `chunk_size` | `100-4000` | Characters per chunk |
| `embedding_model` | Any Hugging Face model | Embedding model |
| `max_chunks_for_context` | `1-50` | Chunks for context |

### 🚨 Troubleshooting

**No transcript available**
- Some videos don't have transcripts
- Try videos with auto-generated captions

**API rate limits**
- Use local models for heavy processing
- Implement delays between requests

**Memory issues**
- Reduce `chunk_size` and `batch_size`
- Use lighter embedding models

**Import errors**
- Run `pip install -r requirements.txt`
- Check Python version (3.9+ required)

### 📚 Examples

See the `examples/` directory:
- `basic_usage.py` - Simple analysis
- `advanced_usage.py` - Batch processing, custom configs
- `notebook_example.py` - Jupyter notebook demo

### 🧪 Testing

```bash
# Run installation test
python test_installation.py

# Run full test suite
poetry run pytest tests/

# Test specific functionality
poetry run python examples/basic_usage.py
```

### 📈 Performance Tips

1. **Use GPU** for local models when available
2. **Adjust chunk sizes** based on content type
3. **Cache embeddings** for repeated analysis
4. **Use Groq API** for fastest cloud inference
5. **Local models** for privacy-sensitive content

### 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### 📄 License

MIT License - see `LICENSE` file for details.

---

**Need help?** Check the full `README.md` or create an issue in the repository.