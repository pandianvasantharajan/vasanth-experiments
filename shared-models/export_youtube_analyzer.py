#!/usr/bin/env python3
"""
Export YouTube Analyzer model to pickle file for API deployment.

This script creates a pickle file containing the YouTubeAnalyzer model
that can be loaded by the model-service API.

Input: YouTube URL
Output: Summary and raw transcript text
"""

import os
import sys
import pickle
from pathlib import Path
from dotenv import load_dotenv

# Add the youtube-video-analyser-model src to path
youtube_model_root = Path(__file__).parent.parent / "youtube-video-analyser-model"
sys.path.insert(0, str(youtube_model_root / "src"))

from youtube_analyzer import YouTubeAnalyzer
from youtube_analyzer.models.schemas import AnalyzerConfig


class YouTubeAnalyzerWrapper:
    """
    Wrapper class for YouTube Analyzer with simplified interface for API use.
    
    Input: YouTube URL (string)
    Output: Dictionary with 'summary' and 'raw_text' keys
    """
    
    def __init__(self, config: AnalyzerConfig = None):
        """Initialize the YouTube Analyzer with configuration."""
        if config is None:
            # Default configuration
            config = AnalyzerConfig(
                embedding_model="all-MiniLM-L6-v2",
                chunking_method="langchain",
                chunk_size=1000,
                chunk_overlap=200,
                llm_provider=os.getenv("LLM_PROVIDER", "groq"),
                chroma_persist_directory="./data/youtube_chroma_db",
                collection_name="youtube_api_transcripts"
            )
        
        self.config = config
        self._analyzer = None
    
    @property
    def analyzer(self):
        """Lazy load analyzer to avoid pickling issues."""
        if self._analyzer is None:
            self._analyzer = YouTubeAnalyzer(config=self.config)
        return self._analyzer
    
    def analyze(self, youtube_url: str) -> dict:
        """
        Analyze a YouTube video and return summary and raw text.
        
        Args:
            youtube_url: YouTube video URL
            
        Returns:
            dict with keys:
                - summary: str (LLM-generated summary)
                - raw_text: str (full transcript text)
                - video_id: str
                - key_points: list[str]
                - metadata: dict (video metadata)
        """
        try:
            # Analyze the video
            result = self.analyzer.analyze_video(youtube_url)
            
            # Extract raw text from context chunks
            raw_text = ""
            if result.context_chunks:
                raw_text = " ".join([chunk.text for chunk in result.context_chunks])
            
            # Build response
            response = {
                "summary": result.summary,
                "raw_text": raw_text,
                "video_id": result.metadata.video_id,
                "key_points": result.key_points or [],
                "metadata": {
                    "title": result.metadata.title,
                    "duration": result.metadata.duration,
                    "language": result.metadata.language,
                    "transcript_length": result.metadata.transcript_length,
                    "chunk_count": result.metadata.chunk_count,
                    "analysis_timestamp": str(result.analysis_timestamp)
                }
            }
            
            return response
            
        except Exception as e:
            raise Exception(f"Error analyzing YouTube video: {str(e)}")
    
    def __call__(self, youtube_url: str) -> dict:
        """Make the class callable."""
        return self.analyze(youtube_url)


def export_model():
    """Export the YouTube Analyzer model to pickle file."""
    
    # Load environment variables
    env_path = youtube_model_root / ".env"
    load_dotenv(env_path)
    
    print("=" * 80)
    print("EXPORTING YOUTUBE ANALYZER MODEL TO PICKLE")
    print("=" * 80)
    
    # Create the wrapper with configuration from environment
    print("\n1. Creating YouTubeAnalyzerWrapper...")
    config = AnalyzerConfig(
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        chunking_method=os.getenv("CHUNKING_METHOD", "langchain"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        llm_provider=os.getenv("LLM_PROVIDER", "groq"),
        chroma_persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/youtube_chroma_db"),
        collection_name="youtube_api_transcripts"
    )
    
    wrapper = YouTubeAnalyzerWrapper(config=config)
    print("   ✓ Wrapper created successfully")
    
    # Create metadata
    metadata = {
        "model_name": "youtube_analyzer",
        "version": "1.0.0",
        "description": "YouTube Video Analyzer with RAG pipeline",
        "input": "youtube_url (str)",
        "output": {
            "summary": "str - LLM-generated summary",
            "raw_text": "str - full transcript text",
            "video_id": "str - YouTube video ID",
            "key_points": "list[str] - extracted key points",
            "metadata": "dict - video metadata"
        },
        "dependencies": [
            "youtube-transcript-api",
            "langchain-text-splitters",
            "sentence-transformers",
            "chromadb",
            "groq",
            "openai"
        ],
        "config": {
            "embedding_model": config.embedding_model,
            "chunking_method": config.chunking_method,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "llm_provider": config.llm_provider
        }
    }
    
    # Package for export
    model_package = {
        "model": wrapper,
        "metadata": metadata
    }
    
    # Export to pickle
    output_path = Path(__file__).parent / "youtube_analyzer.pkl"
    print(f"\n2. Saving model to: {output_path}")
    
    with open(output_path, "wb") as f:
        pickle.dump(model_package, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("   ✓ Model saved successfully")
    
    # Update model registry
    print("\n3. Updating model registry...")
    update_model_registry(output_path, metadata)
    print("   ✓ Registry updated")
    
    print("\n" + "=" * 80)
    print("✅ EXPORT COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"\nModel file: {output_path}")
    print(f"Model size: {output_path.stat().st_size / 1024:.2f} KB")
    print("\nUsage in API:")
    print("  1. Load: model_data = pickle.load(open('youtube_analyzer.pkl', 'rb'))")
    print("  2. Get model: analyzer = model_data['model']")
    print("  3. Use: result = analyzer.analyze('https://youtube.com/watch?v=...')")
    print("=" * 80)
    
    return output_path


def update_model_registry(model_path: Path, metadata: dict):
    """Update the model registry JSON file."""
    import json
    
    registry_path = Path(__file__).parent / "model_registry.json"
    
    # Load existing registry
    if registry_path.exists():
        try:
            with open(registry_path, "r") as f:
                registry = json.load(f)
            # Ensure models key exists
            if "models" not in registry:
                registry["models"] = []
        except json.JSONDecodeError:
            registry = {"models": []}
    else:
        registry = {"models": []}
    
    # Add/update YouTube analyzer entry
    youtube_entry = {
        "name": "youtube_analyzer",
        "version": metadata["version"],
        "file": str(model_path.name),
        "path": str(model_path.absolute()),
        "description": metadata["description"],
        "input": metadata["input"],
        "output": metadata["output"],
        "dependencies": metadata["dependencies"],
        "config": metadata["config"],
        "exported_at": str(Path(model_path).stat().st_mtime)
    }
    
    # Remove old entry if exists
    registry["models"] = [m for m in registry.get("models", []) if m.get("name") != "youtube_analyzer"]
    
    # Add new entry
    registry["models"].append(youtube_entry)
    
    # Save registry
    with open(registry_path, "w") as f:
        json.dump(registry, indent=2, fp=f)


def test_model(model_path: Path):
    """Test the exported model."""
    print("\n" + "=" * 80)
    print("TESTING EXPORTED MODEL")
    print("=" * 80)
    
    print("\n1. Loading model from pickle...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    analyzer = model_data["model"]
    metadata = model_data["metadata"]
    
    print("   ✓ Model loaded successfully")
    print(f"\n   Model: {metadata['model_name']} v{metadata['version']}")
    print(f"   Description: {metadata['description']}")
    
    print("\n2. Testing with sample YouTube URL...")
    test_url = "https://www.youtube.com/watch?v=fLeJJPxua3E"
    print(f"   URL: {test_url}")
    
    try:
        result = analyzer(test_url)
        
        print("\n   ✓ Analysis completed successfully")
        print(f"\n   Video ID: {result['video_id']}")
        print(f"   Summary length: {len(result['summary'])} characters")
        print(f"   Raw text length: {len(result['raw_text'])} characters")
        print(f"   Key points: {len(result['key_points'])} points")
        print(f"\n   Summary preview:")
        print(f"   {result['summary'][:200]}...")
        
        return True
    except Exception as e:
        print(f"\n   ✗ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Export the model
    model_path = export_model()
    
    # Test the exported model
    print("\n")
    test_model(model_path)
