"""
YouTube Analyzer Wrapper for API deployment
"""
import os
import sys
from pathlib import Path
from typing import Optional

# Add youtube-video-analyser-model to path
youtube_model_path = Path(__file__).parent.parent / "youtube-video-analyser-model" / "src"
if str(youtube_model_path) not in sys.path:
    sys.path.insert(0, str(youtube_model_path))

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
