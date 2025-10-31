#!/usr/bin/env python3
"""
Installation test script for YouTube Video Analyzer.
"""

import sys
import importlib
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test main package import
        import youtube_analyzer
        print("✓ youtube_analyzer imported successfully")
        
        # Test main classes
        from youtube_analyzer import YouTubeAnalyzer, AnalyzerConfig
        print("✓ Main classes imported successfully")
        
        # Test models
        from youtube_analyzer.models.schemas import AnalysisResult, VideoMetadata
        print("✓ Models imported successfully")
        
        # Test core components
        from youtube_analyzer.core import TranscriptFetcher
        print("✓ Core components imported successfully")
        
        # Test utilities
        from youtube_analyzer.utils import validate_youtube_url, format_duration
        print("✓ Utilities imported successfully")
        
        # Test configuration
        from youtube_analyzer.config import get_settings
        print("✓ Configuration imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")
    
    try:
        from youtube_analyzer.utils import validate_youtube_url, format_duration
        
        # Test URL validation
        assert validate_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert not validate_youtube_url("https://example.com")
        print("✓ URL validation works")
        
        # Test duration formatting
        assert format_duration(30) == "30.0s"
        assert format_duration(90) == "1m 30.0s"
        print("✓ Duration formatting works")
        
        # Test config creation
        from youtube_analyzer import AnalyzerConfig
        config = AnalyzerConfig()
        assert config.llm_provider == "groq"
        assert config.chunk_size == 1000
        print("✓ Configuration creation works")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_optional_dependencies():
    """Test which optional dependencies are available."""
    print("\nChecking optional dependencies...")
    
    dependencies = {
        "youtube_transcript_api": "YouTube transcript fetching",
        "langchain": "LangChain text splitting",
        "spacy": "spaCy text processing",
        "chromadb": "ChromaDB vector storage",
        "sentence_transformers": "Sentence transformers embeddings",
        "groq": "Groq API client",
        "openai": "OpenAI API client",
        "transformers": "HuggingFace transformers",
        "torch": "PyTorch",
    }
    
    available = []
    missing = []
    
    for dep, description in dependencies.items():
        try:
            importlib.import_module(dep)
            print(f"✓ {dep} - {description}")
            available.append(dep)
        except ImportError:
            print(f"✗ {dep} - {description} (not installed)")
            missing.append(dep)
    
    print(f"\nSummary: {len(available)}/{len(dependencies)} dependencies available")
    
    if missing:
        print("\nTo install missing dependencies:")
        print("pip install", " ".join(missing))
    
    return len(available) > 0


def main():
    """Run all tests."""
    print("YouTube Video Analyzer - Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test basic functionality
    if not test_basic_functionality():
        all_passed = False
    
    # Test optional dependencies
    if not test_optional_dependencies():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! The package is ready to use.")
        print("\nNext steps:")
        print("1. Copy .env.example to .env and add your API keys")
        print("2. Run: python examples/basic_usage.py")
        print("3. Or use the CLI: python -m youtube_analyzer.cli analyze <youtube_url>")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Make sure you're in the project directory")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Or use Poetry: poetry install")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())