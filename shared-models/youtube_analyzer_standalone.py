#!/usr/bin/env python3
"""
YouTube Video Analyzer - Standalone Service Module
==================================================

Complete standalone implementation of the YouTube Video Analyzer for service integration.
This module contains all necessary classes and functions without external dependencies.

Usage:
    from youtube_analyzer_standalone import YouTubeAnalyzerService
    
    service = YouTubeAnalyzerService()
    result = service.analyze_video("https://youtube.com/watch?v=...")
"""

import os
import logging
import json
import re
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import time
from functools import wraps

# Data handling
import pandas as pd
import numpy as np
from tqdm import tqdm

# YouTube and transcript handling
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ML/NLP libraries
from sentence_transformers import SentenceTransformer
import torch

# LLM providers
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configuration and validation
from pydantic import BaseModel, Field
import warnings
warnings.filterwarnings('ignore')


class VideoMetadata(BaseModel):
    """Metadata for a YouTube video."""
    video_id: str
    title: Optional[str] = None
    duration: Optional[float] = None
    chunk_count: Optional[int] = None
    language: Optional[str] = None


class ChunkData(BaseModel):
    """Data structure for text chunks."""
    video_id: str
    chunk_id: str
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    chunk_index: int
    embedding: Optional[List[float]] = None


class AnalysisResult(BaseModel):
    """Result of video analysis."""
    summary: str
    key_points: List[str] = []
    metadata: VideoMetadata
    context_chunks: List[ChunkData] = []
    analysis_timestamp: datetime = Field(default_factory=datetime.now)


class AnalyzerConfig(BaseModel):
    """Configuration for YouTube Analyzer."""
    llm_provider: str = "groq"
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    chunking_method: str = "langchain"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunks_for_context: int = 8
    chroma_persist_directory: str = "./data/chroma_db"
    collection_name: str = "youtube_videos"
    batch_size: int = 32
    max_retries: int = 3
    request_timeout: int = 30


class TranscriptFetcher:
    """Handles YouTube transcript fetching and processing."""
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        if 'youtu.be/' in url:
            return url.split('youtu.be/')[-1].split('?')[0]
        elif 'watch?v=' in url:
            return url.split('watch?v=')[-1].split('&')[0]
        elif len(url) == 11:  # Direct video ID
            return url
        else:
            raise ValueError(f"Invalid YouTube URL: {url}")
    
    def fetch_video_data(self, url: str) -> tuple:
        """Fetch transcript and metadata for a YouTube video."""
        try:
            video_id = self.extract_video_id(url)
            
            # Get transcript
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Create transcript segments
            segments = []
            for item in transcript_list:
                segment = type('Segment', (), {
                    'text': item['text'],
                    'start': item['start'],
                    'duration': item['duration']
                })()
                segments.append(segment)
            
            # Calculate total duration
            total_duration = max([seg.start + seg.duration for seg in segments]) if segments else 0
            
            # Create metadata
            metadata = VideoMetadata(
                video_id=video_id,
                duration=total_duration,
                language='en'
            )
            
            return segments, metadata
            
        except Exception as e:
            raise Exception(f"Failed to fetch transcript for {url}: {str(e)}")


class LangChainChunker:
    """LangChain-based text chunker."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_text(self, text: str, video_id: str) -> List[ChunkData]:
        """Chunk text using LangChain splitter."""
        chunks = self.splitter.split_text(text)
        
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append(ChunkData(
                video_id=video_id,
                chunk_id=f"{video_id}_chunk_{i}",
                text=chunk.strip(),
                chunk_index=i
            ))
        
        return chunk_data


class Embedder:
    """Handles text embedding generation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
    
    def _initialize_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                raise Exception(f"Failed to load embedding model: {e}")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        self._initialize_model()
        
        try:
            embeddings = self.model.encode(texts, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            raise Exception(f"Failed to generate embeddings: {e}")
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]


class VectorRetriever:
    """Handles vector storage and retrieval using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "youtube_videos"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        if self.client is None:
            try:
                import chromadb
                
                # Create directory if it doesn't exist
                os.makedirs(self.persist_directory, exist_ok=True)
                
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                
                # Get or create collection
                try:
                    self.collection = self.client.get_collection(name=self.collection_name)
                except:
                    self.collection = self.client.create_collection(name=self.collection_name)
                    
            except Exception as e:
                raise Exception(f"Failed to initialize ChromaDB: {e}")
    
    def store_chunks(self, chunks: List[ChunkData], embeddings: List[List[float]]):
        """Store chunks and their embeddings in the vector database."""
        self._initialize_client()
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks]
            documents = [chunk.text for chunk in chunks]
            metadatas = [
                {
                    "video_id": chunk.video_id,
                    "chunk_index": chunk.chunk_index,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time
                }
                for chunk in chunks
            ]
            
            # Add to collection
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
        except Exception as e:
            raise Exception(f"Failed to store chunks: {e}")
    
    def search_similar(self, query_embedding: List[float], video_id: Optional[str] = None, 
                      max_results: int = 10) -> List[ChunkData]:
        """Search for similar chunks using vector similarity."""
        self._initialize_client()
        
        try:
            # Prepare search parameters
            where_filter = {"video_id": video_id} if video_id else None
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_filter
            )
            
            # Convert results to ChunkData objects
            chunk_results = []
            for i in range(len(results['ids'][0])):
                chunk_data = ChunkData(
                    video_id=results['metadatas'][0][i]['video_id'],
                    chunk_id=results['ids'][0][i],
                    text=results['documents'][0][i],
                    chunk_index=results['metadatas'][0][i]['chunk_index'],
                    start_time=results['metadatas'][0][i].get('start_time'),
                    end_time=results['metadatas'][0][i].get('end_time')
                )
                chunk_results.append(chunk_data)
            
            return chunk_results
            
        except Exception as e:
            raise Exception(f"Failed to search similar chunks: {e}")


class GroqSummarizer:
    """Groq-based summarizer."""
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.api_key = api_key
        self.model = model
        self.client = None
    
    def _initialize_client(self):
        """Initialize Groq client."""
        if self.client is None and GROQ_AVAILABLE:
            self.client = Groq(api_key=self.api_key)
    
    def summarize(self, text: str, query: Optional[str] = None) -> str:
        """Generate summary using Groq."""
        if not GROQ_AVAILABLE:
            return "Groq not available. Please install: pip install groq"
        
        self._initialize_client()
        
        # Create prompt
        if query:
            prompt = f"""Based on the following text, provide a focused summary that addresses this query: "{query}"

Text: {text}

Summary:"""
        else:
            prompt = f"""Please provide a comprehensive summary of the following text:

{text}

Summary:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            return f"Error generating summary: {str(e)}"


class YouTubeAnalyzerService:
    """
    Complete YouTube Video Analyzer Service for API integration.
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """Initialize the YouTube analyzer service."""
        
        # Use default config if none provided
        if config is None:
            config = AnalyzerConfig()
        
        self.config = config
        
        # Initialize components
        self.transcript_fetcher = TranscriptFetcher()
        self.chunker = LangChainChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embedder = Embedder(model_name=config.embedding_model)
        self.retriever = VectorRetriever(
            persist_directory=config.chroma_persist_directory,
            collection_name=config.collection_name
        )
        
        # Initialize summarizer if API key available
        self.summarizer = None
        if config.groq_api_key and GROQ_AVAILABLE:
            self.summarizer = GroqSummarizer(config.groq_api_key)
    
    def analyze_video(self, url: str, query: Optional[str] = None, 
                     max_chunks: Optional[int] = None) -> AnalysisResult:
        """
        Analyze a YouTube video and return structured results.
        
        Args:
            url: YouTube URL or video ID
            query: Optional specific query for focused analysis
            max_chunks: Maximum number of chunks to use for context
        
        Returns:
            AnalysisResult with summary, key points, and metadata
        """
        
        try:
            # Step 1: Extract and fetch transcript
            video_id = self.transcript_fetcher.extract_video_id(url)
            segments, metadata = self.transcript_fetcher.fetch_video_data(url)
            full_transcript = ' '.join([seg.text for seg in segments])
            
            # Step 2: Intelligent text chunking
            chunks = self.chunker.chunk_text(full_transcript, video_id)
            metadata.chunk_count = len(chunks)
            
            # Step 3: Generate semantic embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = self.embedder.generate_embeddings(chunk_texts)
            
            # Step 4: Store in vector database
            self.retriever.store_chunks(chunks, embeddings)
            
            # Step 5: Intelligent chunk retrieval
            search_query = query or "main topics, key points, and important insights from this video"
            query_embedding = self.embedder.generate_single_embedding(search_query)
            
            max_chunks = max_chunks or self.config.max_chunks_for_context
            relevant_chunks = self.retriever.search_similar(
                query_embedding, video_id=video_id, max_results=max_chunks
            )
            
            # Step 6: AI-powered summarization
            if self.summarizer:
                # Combine most relevant chunks
                context_text = "\n\n".join([chunk.text for chunk in relevant_chunks])
                summary = self.summarizer.summarize(context_text, query)
                
                # Extract key insights
                key_points = self._extract_key_insights(summary)
            else:
                summary = "Analysis complete. For AI-powered summarization, please configure GROQ_API_KEY."
                key_points = ["Configure API keys for enhanced analysis"]
            
            # Create comprehensive result
            result = AnalysisResult(
                summary=summary,
                key_points=key_points,
                metadata=metadata,
                context_chunks=relevant_chunks[:5],  # Top 5 most relevant
                analysis_timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Video analysis failed: {str(e)}")
    
    def _extract_key_insights(self, summary: str) -> List[str]:
        """Extract key insights from the summary."""
        if not summary or len(summary) < 50:
            return ["Analysis too brief to extract key points"]
        
        sentences = summary.split('. ')
        insights = []
        
        # Look for important sentences
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 25:  # Substantial content
                # Priority keywords for key insights
                if any(keyword in sentence.lower() for keyword in 
                      ['key', 'important', 'main', 'primary', 'significant', 'crucial', 
                       'highlights', 'focuses', 'discusses', 'explains', 'demonstrates']):
                    insights.append(sentence)
                elif len(insights) < 3:  # Ensure we have some insights
                    insights.append(sentence)
        
        return insights[:5] if insights else ["Summary analysis completed"]


# Service factory function
def create_youtube_analyzer_service(groq_api_key: Optional[str] = None, 
                                  openai_api_key: Optional[str] = None) -> YouTubeAnalyzerService:
    """
    Factory function to create a YouTube analyzer service with configuration.
    """
    
    # Get API keys from environment if not provided
    groq_key = groq_api_key or os.getenv('GROQ_API_KEY')
    openai_key = openai_api_key or os.getenv('OPENAI_API_KEY')
    
    config = AnalyzerConfig(
        groq_api_key=groq_key,
        openai_api_key=openai_key,
        llm_provider="groq" if groq_key else "local"
    )
    
    return YouTubeAnalyzerService(config)