"""
YouTube Video Analyser Model

A robust RAG pipeline for summarizing YouTube videos using transcript analysis.
"""

from .analyzer import YouTubeAnalyzer
from .models.schemas import AnalysisResult, AnalyzerConfig
from .config.settings import Settings

__version__ = "0.1.0"
__all__ = ["YouTubeAnalyzer", "AnalysisResult", "AnalyzerConfig", "Settings"]