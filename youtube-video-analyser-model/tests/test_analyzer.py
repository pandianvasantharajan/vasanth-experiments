"""
Test suite for YouTube Video Analyzer.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add src to path for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from youtube_analyzer import YouTubeAnalyzer, AnalyzerConfig
from youtube_analyzer.models.schemas import TranscriptSegment, VideoMetadata
from youtube_analyzer.utils import validate_youtube_url, format_duration


class TestYouTubeURLValidation:
    """Test YouTube URL validation."""
    
    def test_valid_youtube_urls(self):
        """Test various valid YouTube URL formats."""
        valid_urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "dQw4w9WgXcQ"  # Just the video ID
        ]
        
        for url in valid_urls:
            assert validate_youtube_url(url), f"URL should be valid: {url}"
    
    def test_invalid_youtube_urls(self):
        """Test invalid YouTube URLs."""
        invalid_urls = [
            "https://vimeo.com/123456",
            "https://www.example.com",
            "not_a_url",
            "https://youtube.com/watch?v=",  # Missing video ID
            "abc123"  # Too short for video ID
        ]
        
        for url in invalid_urls:
            assert not validate_youtube_url(url), f"URL should be invalid: {url}"


class TestDurationFormatting:
    """Test duration formatting utility."""
    
    def test_seconds_formatting(self):
        """Test formatting of durations in seconds."""
        assert format_duration(30.5) == "30.5s"
        assert format_duration(45) == "45.0s"
    
    def test_minutes_formatting(self):
        """Test formatting of durations in minutes."""
        assert format_duration(90) == "1m 30.0s"
        assert format_duration(125.5) == "2m 5.5s"
    
    def test_hours_formatting(self):
        """Test formatting of durations in hours."""
        assert format_duration(3661) == "1h 1m 1.0s"
        assert format_duration(7200) == "2h 0m 0.0s"


class TestTranscriptSegment:
    """Test TranscriptSegment model."""
    
    def test_segment_creation(self):
        """Test creating transcript segments."""
        segment = TranscriptSegment(
            text="Hello world",
            start=10.0,
            duration=5.0
        )
        
        assert segment.text == "Hello world"
        assert segment.start == 10.0
        assert segment.duration == 5.0
        assert segment.end == 15.0
    
    def test_end_time_calculation(self):
        """Test end time calculation."""
        segment = TranscriptSegment(
            text="Test",
            start=100.5,
            duration=30.2
        )
        
        assert segment.end == 130.7


class TestAnalyzerConfig:
    """Test AnalyzerConfig model."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AnalyzerConfig()
        
        assert config.llm_provider == "groq"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunking_method == "langchain"
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = AnalyzerConfig(
            llm_provider="openai",
            chunk_size=500,
            chunking_method="spacy"
        )
        
        assert config.llm_provider == "openai"
        assert config.chunk_size == 500
        assert config.chunking_method == "spacy"
        # Other values should be defaults
        assert config.embedding_model == "all-MiniLM-L6-v2"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config should not raise
        config = AnalyzerConfig(chunk_size=500, chunk_overlap=100)
        
        # Invalid config should raise
        with pytest.raises(ValueError):
            AnalyzerConfig(chunk_size=0)
        
        with pytest.raises(ValueError):
            AnalyzerConfig(chunk_size=100, chunk_overlap=150)


@pytest.fixture
def mock_transcript_data():
    """Fixture providing mock transcript data."""
    return [
        {"text": "Hello everyone", "start": 0.0, "duration": 2.0},
        {"text": "Welcome to this video", "start": 2.0, "duration": 3.0},
        {"text": "Today we'll discuss", "start": 5.0, "duration": 2.5},
        {"text": "artificial intelligence", "start": 7.5, "duration": 3.0},
        {"text": "and machine learning", "start": 10.5, "duration": 2.8}
    ]


@pytest.fixture
def temp_db_dir():
    """Fixture providing temporary database directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestTranscriptFetcher:
    """Test transcript fetching functionality."""
    
    @patch('youtube_analyzer.core.transcript_fetcher.YouTubeTranscriptApi')
    def test_extract_video_id(self, mock_api):
        """Test video ID extraction from URLs."""
        from youtube_analyzer.core.transcript_fetcher import TranscriptFetcher
        
        fetcher = TranscriptFetcher()
        
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ")
        ]
        
        for url, expected_id in test_cases:
            assert fetcher.extract_video_id(url) == expected_id
    
    @patch('youtube_analyzer.core.transcript_fetcher.YouTubeTranscriptApi')
    def test_fetch_transcript_segments(self, mock_api, mock_transcript_data):
        """Test fetching transcript segments."""
        from youtube_analyzer.core.transcript_fetcher import TranscriptFetcher
        
        # Mock the API response
        mock_api.get_transcript.return_value = mock_transcript_data
        
        fetcher = TranscriptFetcher()
        segments = fetcher.fetch_transcript_segments("test_video_id")
        
        assert len(segments) == 5
        assert segments[0].text == "Hello everyone"
        assert segments[0].start == 0.0
        assert segments[0].duration == 2.0


class TestChunking:
    """Test text chunking functionality."""
    
    def test_langchain_chunker(self):
        """Test LangChain chunking."""
        # Skip if LangChain not available
        pytest.importorskip("langchain")
        
        from youtube_analyzer.core.chunker import LangChainChunker
        
        chunker = LangChainChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is a test text. " * 10  # Repeat to make it longer
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk) <= 60 for chunk in chunks)  # Allow some flexibility
    
    def test_spacy_chunker(self):
        """Test spaCy chunking."""
        # Skip if spaCy not available
        pytest.importorskip("spacy")
        
        from youtube_analyzer.core.chunker import SpacyChunker
        
        chunker = SpacyChunker(chunk_size=50, chunk_overlap=10)
        
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, str) for chunk in chunks)


class TestEmbedding:
    """Test embedding functionality."""
    
    @patch('youtube_analyzer.core.embedder.SentenceTransformer')
    def test_embedder_initialization(self, mock_transformer):
        """Test embedder initialization."""
        from youtube_analyzer.core.embedder import Embedder
        
        # Mock the transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_transformer.return_value = mock_model
        
        embedder = Embedder("test-model")
        
        assert embedder.model_name == "test-model"
        assert embedder.get_dimension() == 384
    
    @patch('youtube_analyzer.core.embedder.SentenceTransformer')
    def test_encode_texts(self, mock_transformer):
        """Test text encoding."""
        from youtube_analyzer.core.embedder import Embedder
        
        # Mock the transformer
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_transformer.return_value = mock_model
        
        embedder = Embedder("test-model")
        
        texts = ["Hello world", "Test text"]
        result = embedder.encode_texts(texts)
        
        assert len(result.embeddings) == 2
        assert len(result.embeddings[0]) == 3
        assert result.model_name == "test-model"
        assert result.dimension == 384


class TestVectorRetriever:
    """Test vector retrieval functionality."""
    
    @patch('youtube_analyzer.core.retriever.chromadb')
    def test_retriever_initialization(self, mock_chromadb, temp_db_dir):
        """Test retriever initialization."""
        from youtube_analyzer.core.retriever import VectorRetriever
        
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        retriever = VectorRetriever(
            persist_directory=temp_db_dir,
            collection_name="test_collection"
        )
        
        assert retriever.persist_directory == temp_db_dir
        assert retriever.collection_name == "test_collection"


class TestIntegration:
    """Integration tests for the full pipeline."""
    
    @patch('youtube_analyzer.core.transcript_fetcher.YouTubeTranscriptApi')
    @patch('youtube_analyzer.core.embedder.SentenceTransformer')
    @patch('youtube_analyzer.core.retriever.chromadb')
    @patch('youtube_analyzer.core.summarizer.Groq')
    def test_full_analysis_pipeline(
        self, 
        mock_groq,
        mock_chromadb,
        mock_transformer,
        mock_transcript_api,
        mock_transcript_data,
        temp_db_dir
    ):
        """Test the full analysis pipeline."""
        # Mock transcript API
        mock_transcript_api.get_transcript.return_value = mock_transcript_data
        
        # Mock embedder
        mock_model = Mock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = [[0.1] * 384, [0.2] * 384]
        mock_transformer.return_value = mock_model
        
        # Mock ChromaDB
        mock_client = Mock()
        mock_collection = Mock()
        mock_collection.query.return_value = {
            "documents": [["test chunk"]],
            "metadatas": [[{"video_id": "test", "chunk_index": 0}]],
            "distances": [[0.1]]
        }
        mock_collection.count.return_value = 1
        mock_client.get_collection.return_value = mock_collection
        mock_chromadb.PersistentClient.return_value = mock_client
        
        # Mock Groq
        mock_groq_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Test summary"
        mock_groq_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_groq_client
        
        # Set up configuration with test directory
        config = AnalyzerConfig(
            groq_api_key="test_key",
            chroma_persist_directory=temp_db_dir,
            chunk_size=100
        )
        
        # Initialize analyzer
        analyzer = YouTubeAnalyzer(config)
        
        # Run analysis
        result = analyzer.analyze_video("https://www.youtube.com/watch?v=test")
        
        # Verify results
        assert result.summary == "Test summary"
        assert result.metadata.video_id == "test"
        assert len(result.key_points) >= 0


if __name__ == "__main__":
    pytest.main([__file__])