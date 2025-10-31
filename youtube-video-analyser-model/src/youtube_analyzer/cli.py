"""
Command Line Interface for YouTube Video Analyzer.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional

from .analyzer import YouTubeAnalyzer
from .models.schemas import AnalyzerConfig
from .config.settings import get_settings


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )


def create_config_from_args(args: argparse.Namespace) -> AnalyzerConfig:
    """Create configuration from command line arguments."""
    settings = get_settings()
    
    return AnalyzerConfig(
        llm_provider=args.llm_provider or settings.llm_provider,
        groq_api_key=args.groq_api_key or settings.groq_api_key,
        openai_api_key=args.openai_api_key or settings.openai_api_key,
        openrouter_api_key=args.openrouter_api_key or settings.openrouter_api_key,
        local_model_name=args.local_model_name or settings.local_model_name,
        embedding_model=args.embedding_model or settings.embedding_model,
        chunking_method=args.chunking_method or settings.chunking_method,
        chunk_size=args.chunk_size or settings.chunk_size,
        chunk_overlap=args.chunk_overlap or settings.chunk_overlap,
        max_chunks_for_context=args.max_chunks or settings.max_chunks_for_context,
        chroma_persist_directory=args.persist_dir or settings.chroma_persist_directory,
        collection_name=args.collection_name or settings.collection_name,
        batch_size=args.batch_size or settings.batch_size,
        max_retries=args.max_retries or settings.max_retries,
        request_timeout=args.request_timeout or settings.request_timeout
    )


def analyze_video_command(args: argparse.Namespace) -> None:
    """Handle video analysis command."""
    try:
        # Setup logging
        setup_logging(args.log_level)
        
        # Create configuration
        config = create_config_from_args(args)
        
        # Initialize analyzer
        analyzer = YouTubeAnalyzer(config)
        
        # Analyze video
        result = analyzer.analyze_video(
            url=args.url,
            include_context=args.include_context,
            max_chunks=args.max_chunks,
            query=args.query
        )
        
        # Output results
        if args.output_format == "json":
            output = {
                "summary": result.summary,
                "key_points": result.key_points,
                "metadata": {
                    "video_id": result.metadata.video_id,
                    "title": result.metadata.title,
                    "duration": result.metadata.duration,
                    "language": result.metadata.language,
                    "transcript_length": result.metadata.transcript_length,
                    "chunk_count": result.metadata.chunk_count
                },
                "analysis_timestamp": result.analysis_timestamp,
                "config_used": result.config_used
            }
            
            if args.include_context and result.context_chunks:
                output["context_chunks"] = [
                    {
                        "text": chunk.text,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "similarity_score": chunk.similarity_score
                    }
                    for chunk in result.context_chunks
                ]
            
            print(json.dumps(output, indent=2))
        else:
            # Human-readable format
            print("=" * 80)
            print("YOUTUBE VIDEO ANALYSIS RESULTS")
            print("=" * 80)
            print(f"Video ID: {result.metadata.video_id}")
            if result.metadata.duration:
                print(f"Duration: {result.metadata.duration:.1f} seconds")
            if result.metadata.language:
                print(f"Language: {result.metadata.language}")
            print(f"Chunks created: {result.metadata.chunk_count}")
            print(f"Analysis time: {result.analysis_timestamp}")
            print()
            
            print("SUMMARY:")
            print("-" * 40)
            print(result.summary)
            print()
            
            if result.key_points:
                print("KEY POINTS:")
                print("-" * 40)
                for i, point in enumerate(result.key_points, 1):
                    print(f"{i}. {point}")
                print()
            
            if args.include_context and result.context_chunks:
                print("RELEVANT CONTEXT:")
                print("-" * 40)
                for i, chunk in enumerate(result.context_chunks, 1):
                    time_info = ""
                    if chunk.start_time is not None and chunk.end_time is not None:
                        time_info = f" [{chunk.start_time:.1f}s-{chunk.end_time:.1f}s]"
                    similarity_info = ""
                    if chunk.similarity_score is not None:
                        similarity_info = f" (similarity: {chunk.similarity_score:.3f})"
                    
                    print(f"{i}.{time_info}{similarity_info}")
                    print(f"   {chunk.text[:200]}...")
                    print()
        
        # Save output to file if specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if args.output_format == "json":
                with open(output_path, 'w') as f:
                    json.dump(output, f, indent=2)
            else:
                # Save as text file
                with open(output_path, 'w') as f:
                    f.write(f"YouTube Video Analysis Results\n")
                    f.write(f"Video ID: {result.metadata.video_id}\n")
                    f.write(f"Analysis Time: {result.analysis_timestamp}\n\n")
                    f.write(f"Summary:\n{result.summary}\n\n")
                    
                    if result.key_points:
                        f.write("Key Points:\n")
                        for i, point in enumerate(result.key_points, 1):
                            f.write(f"{i}. {point}\n")
            
            print(f"Results saved to: {output_path}")
            
    except Exception as e:
        print(f"Error analyzing video: {e}", file=sys.stderr)
        sys.exit(1)


def transcript_command(args: argparse.Namespace) -> None:
    """Handle transcript extraction command."""
    try:
        setup_logging(args.log_level)
        
        config = create_config_from_args(args)
        analyzer = YouTubeAnalyzer(config)
        
        transcript = analyzer.get_transcript(args.url)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                f.write(transcript)
            print(f"Transcript saved to: {args.output_file}")
        else:
            print(transcript)
            
    except Exception as e:
        print(f"Error extracting transcript: {e}", file=sys.stderr)
        sys.exit(1)


def search_command(args: argparse.Namespace) -> None:
    """Handle search command."""
    try:
        setup_logging(args.log_level)
        
        config = create_config_from_args(args)
        analyzer = YouTubeAnalyzer(config)
        
        results = analyzer.search_similar_content(
            query=args.query,
            video_id=args.video_id,
            max_results=args.max_results
        )
        
        if args.output_format == "json":
            output = [
                {
                    "text": chunk.text,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "similarity_score": chunk.similarity_score
                }
                for chunk in results
            ]
            print(json.dumps(output, indent=2))
        else:
            print(f"Found {len(results)} similar chunks:")
            print("-" * 40)
            for i, chunk in enumerate(results, 1):
                time_info = ""
                if chunk.start_time is not None and chunk.end_time is not None:
                    time_info = f" [{chunk.start_time:.1f}s-{chunk.end_time:.1f}s]"
                similarity_info = ""
                if chunk.similarity_score is not None:
                    similarity_info = f" (similarity: {chunk.similarity_score:.3f})"
                
                print(f"{i}.{time_info}{similarity_info}")
                print(f"   {chunk.text}")
                print()
                
    except Exception as e:
        print(f"Error searching: {e}", file=sys.stderr)
        sys.exit(1)


def stats_command(args: argparse.Namespace) -> None:
    """Handle stats command."""
    try:
        setup_logging(args.log_level)
        
        config = create_config_from_args(args)
        analyzer = YouTubeAnalyzer(config)
        
        stats = analyzer.get_stats()
        
        if args.output_format == "json":
            print(json.dumps(stats, indent=2))
        else:
            print("Database Statistics:")
            print("-" * 30)
            for key, value in stats.items():
                print(f"{key.replace('_', ' ').title()}: {value}")
                
    except Exception as e:
        print(f"Error getting stats: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Video Analyzer - RAG pipeline for video summarization",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Global options
    parser.add_argument("--log-level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--output-format", default="text", 
                       choices=["text", "json"],
                       help="Output format")
    
    # Configuration options
    parser.add_argument("--llm-provider", choices=["groq", "openai", "openrouter", "local"],
                       help="LLM provider to use")
    parser.add_argument("--groq-api-key", help="Groq API key")
    parser.add_argument("--openai-api-key", help="OpenAI API key")
    parser.add_argument("--openrouter-api-key", help="OpenRouter API key")
    parser.add_argument("--local-model-name", help="Local model name")
    parser.add_argument("--embedding-model", help="Embedding model name")
    parser.add_argument("--chunking-method", choices=["langchain", "spacy"],
                       help="Text chunking method")
    parser.add_argument("--chunk-size", type=int, help="Chunk size")
    parser.add_argument("--chunk-overlap", type=int, help="Chunk overlap")
    parser.add_argument("--max-chunks", type=int, help="Max chunks for context")
    parser.add_argument("--persist-dir", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", help="ChromaDB collection name")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    parser.add_argument("--max-retries", type=int, help="Maximum retries")
    parser.add_argument("--request-timeout", type=int, help="Request timeout")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a YouTube video")
    analyze_parser.add_argument("url", help="YouTube URL or video ID")
    analyze_parser.add_argument("--query", help="Specific query for focused analysis")
    analyze_parser.add_argument("--include-context", action="store_true", default=True,
                                help="Include context chunks in output")
    analyze_parser.add_argument("--output-file", help="Save output to file")
    analyze_parser.set_defaults(func=analyze_video_command)
    
    # Transcript command
    transcript_parser = subparsers.add_parser("transcript", help="Extract transcript only")
    transcript_parser.add_argument("url", help="YouTube URL or video ID")
    transcript_parser.add_argument("--output-file", help="Save transcript to file")
    transcript_parser.set_defaults(func=transcript_command)
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search for similar content")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--video-id", help="Filter by video ID")
    search_parser.add_argument("--max-results", type=int, default=10,
                              help="Maximum number of results")
    search_parser.set_defaults(func=search_command)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=stats_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Execute command
    args.func(args)


if __name__ == "__main__":
    main()