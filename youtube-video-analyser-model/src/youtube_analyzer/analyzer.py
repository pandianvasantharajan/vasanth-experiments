"""
Main YouTube Analyzer class that orchestrates the RAG pipeline.
"""

import logging
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from .core import (
    TranscriptFetcher,
    ChunkerFactory,
    Embedder,
    VectorRetriever,
    SummarizerFactory
)
from .models.schemas import (
    AnalysisResult,
    AnalyzerConfig,
    VideoMetadata,
    ChunkData
)
from .config.settings import get_settings

logger = logging.getLogger(__name__)


class YouTubeAnalyzer:
    """
    Main YouTube video analyzer using RAG pipeline.
    
    This class orchestrates the entire process:
    1. Fetch YouTube transcript
    2. Chunk the text semantically
    3. Generate embeddings
    4. Store in vector database
    5. Retrieve relevant chunks
    6. Generate summary using LLM
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        Initialize the YouTube analyzer.
        
        Args:
            config: Optional configuration object. If None, loads from environment.
        """
        # Load configuration
        if config is None:
            settings = get_settings()
            config = AnalyzerConfig(
                llm_provider=settings.llm_provider,
                groq_api_key=settings.groq_api_key,
                openai_api_key=settings.openai_api_key,
                openrouter_api_key=settings.openrouter_api_key,
                local_model_name=settings.local_model_name,
                embedding_model=settings.embedding_model,
                chunking_method=settings.chunking_method,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                max_chunks_for_context=settings.max_chunks_for_context,
                chroma_persist_directory=settings.chroma_persist_directory,
                collection_name=settings.collection_name,
                batch_size=settings.batch_size,
                max_retries=settings.max_retries,
                request_timeout=settings.request_timeout
            )
        
        self.config = config
        
        # Initialize components
        self.transcript_fetcher = TranscriptFetcher()
        self.chunker = ChunkerFactory.create_chunker(
            method=config.chunking_method,
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        self.embedder = Embedder(model_name=config.embedding_model)
        self.retriever = VectorRetriever(
            persist_directory=config.chroma_persist_directory,
            collection_name=config.collection_name
        )
        
        # Initialize summarizer based on provider
        self.summarizer = SummarizerFactory.create_summarizer(
            provider=config.llm_provider,
            groq_api_key=config.groq_api_key,
            openai_api_key=config.openai_api_key,
            openrouter_api_key=config.openrouter_api_key,
            local_model_name=config.local_model_name
        )
        
        logger.info("YouTube Analyzer initialized successfully")
    
    def analyze_video(
        self,
        url: str,
        include_context: bool = True,
        max_chunks: Optional[int] = None,
        query: Optional[str] = None
    ) -> AnalysisResult:
        """
        Analyze a YouTube video and generate a summary.
        
        Args:
            url: YouTube URL or video ID
            include_context: Whether to include context chunks in result
            max_chunks: Maximum number of chunks to use for context
            query: Optional specific query for focused analysis
            
        Returns:
            AnalysisResult with summary and metadata
        """
        try:
            logger.info(f"Starting analysis of video: {url}")
            
            # Step 1: Fetch transcript
            video_id = self.transcript_fetcher.extract_video_id(url)
            segments, metadata = self.transcript_fetcher.fetch_video_data(url)
            full_transcript = ' '.join([seg.text for seg in segments])
            
            logger.info(f"Fetched transcript with {len(segments)} segments")
            
            # Step 2: Chunk the transcript
            chunks = self.chunker.chunk_segments(segments)
            metadata.chunk_count = len(chunks)
            
            logger.info(f"Created {len(chunks)} chunks")
            
            # Step 3: Generate embeddings
            chunk_texts = [chunk.text for chunk in chunks]
            embedding_result = self.embedder.encode_texts(
                chunk_texts,
                batch_size=self.config.batch_size
            )
            
            logger.info(f"Generated embeddings for {len(chunk_texts)} chunks")
            
            # Step 4: Store in vector database
            self.retriever.add_chunks(chunks, embedding_result.embeddings, video_id)
            
            logger.info("Stored chunks in vector database")
            
            # Step 5: Retrieve relevant chunks for context
            context_chunks = None
            if include_context or query:
                # Use query if provided, otherwise use a general summary query
                search_query = query or "main topics key points summary important information"
                query_embedding = self.embedder.encode_single(search_query)
                
                max_results = max_chunks or self.config.max_chunks_for_context
                retrieval_result = self.retriever.search_similar_chunks(
                    query_embedding=query_embedding,
                    n_results=max_results,
                    video_id=video_id
                )
                
                context_chunks = retrieval_result.chunks
                logger.info(f"Retrieved {len(context_chunks)} relevant chunks")
            
            # Step 6: Generate summary
            summary = self.summarizer.summarize(
                text=full_transcript,
                context_chunks=context_chunks
            )
            
            # Extract key points
            key_points = self.summarizer.extract_key_points(full_transcript)
            
            logger.info("Generated summary and key points")
            
            # Create result
            result = AnalysisResult(
                summary=summary,
                key_points=key_points,
                context_chunks=context_chunks if include_context else None,
                metadata=metadata,
                analysis_timestamp=datetime.now().isoformat(),
                config_used={
                    "llm_provider": self.config.llm_provider,
                    "embedding_model": self.config.embedding_model,
                    "chunking_method": self.config.chunking_method,
                    "chunk_size": self.config.chunk_size,
                    "chunk_overlap": self.config.chunk_overlap
                }
            )
            
            logger.info("Analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing video {url}: {e}")
            raise
    
    def get_transcript(self, url: str) -> str:
        """
        Get the transcript text for a video.
        
        Args:
            url: YouTube URL or video ID
            
        Returns:
            Full transcript text
        """
        try:
            video_id = self.transcript_fetcher.extract_video_id(url)
            return self.transcript_fetcher.fetch_transcript_text(video_id)
        except Exception as e:
            logger.error(f"Error getting transcript for {url}: {e}")
            raise
    
    def create_embeddings(self, chunks: List[str]) -> None:
        """
        Create and store embeddings for text chunks.
        
        Args:
            chunks: List of text chunks
        """
        try:
            embedding_result = self.embedder.encode_texts(
                chunks,
                batch_size=self.config.batch_size
            )
            
            # Convert to ChunkData objects
            chunk_data = [
                ChunkData(text=text, chunk_index=i)
                for i, text in enumerate(chunks)
            ]
            
            # Store in database (using dummy video ID)
            self.retriever.add_chunks(
                chunk_data,
                embedding_result.embeddings,
                "manual_chunks"
            )
            
            logger.info(f"Created and stored embeddings for {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def search_similar_content(
        self,
        query: str,
        video_id: Optional[str] = None,
        max_results: int = 10
    ) -> List[ChunkData]:
        """
        Search for content similar to a query.
        
        Args:
            query: Search query
            video_id: Optional video ID to filter results
            max_results: Maximum number of results
            
        Returns:
            List of similar chunks
        """
        try:
            query_embedding = self.embedder.encode_single(query)
            retrieval_result = self.retriever.search_similar_chunks(
                query_embedding=query_embedding,
                n_results=max_results,
                video_id=video_id
            )
            
            return retrieval_result.chunks
            
        except Exception as e:
            logger.error(f"Error searching similar content: {e}")
            raise
    
    def summarize(self, query: str, context: str) -> str:
        """
        Generate a summary based on query and context.
        
        Args:
            query: The query or question
            context: Context text to base the summary on
            
        Returns:
            Generated summary
        """
        try:
            # Create a focused prompt
            focused_text = f"Query: {query}\n\nContext: {context}"
            return self.summarizer.summarize(focused_text)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            raise
    
    def get_video_chunks(self, video_id: str) -> List[ChunkData]:
        """
        Get all chunks for a specific video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of chunks for the video
        """
        try:
            return self.retriever.get_chunks_by_video(video_id)
        except Exception as e:
            logger.error(f"Error getting video chunks: {e}")
            raise
    
    def delete_video_data(self, video_id: str) -> None:
        """
        Delete all data for a specific video.
        
        Args:
            video_id: YouTube video ID
        """
        try:
            self.retriever.delete_video_chunks(video_id)
            logger.info(f"Deleted data for video {video_id}")
        except Exception as e:
            logger.error(f"Error deleting video data: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the analyzer's database.
        
        Returns:
            Dictionary with statistics
        """
        try:
            return self.retriever.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}