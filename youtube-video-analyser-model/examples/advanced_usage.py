"""
Advanced usage examples for YouTube Video Analyzer.
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Any

from youtube_analyzer import YouTubeAnalyzer, AnalyzerConfig


class BatchAnalyzer:
    """Batch analyzer for processing multiple videos."""
    
    def __init__(self, config: AnalyzerConfig):
        self.analyzer = YouTubeAnalyzer(config)
        self.results = []
    
    def analyze_playlist(self, video_urls: List[str]) -> List[Dict[str, Any]]:
        """Analyze multiple videos and collect results."""
        for i, url in enumerate(video_urls, 1):
            try:
                print(f"Analyzing video {i}/{len(video_urls)}: {url}")
                result = self.analyzer.analyze_video(url)
                
                self.results.append({
                    "url": url,
                    "video_id": result.metadata.video_id,
                    "summary": result.summary,
                    "key_points": result.key_points,
                    "duration": result.metadata.duration,
                    "chunk_count": result.metadata.chunk_count,
                    "analysis_timestamp": result.analysis_timestamp
                })
                
                print(f"✓ Completed analysis of {result.metadata.video_id}")
                
            except Exception as e:
                print(f"✗ Failed to analyze {url}: {e}")
                self.results.append({
                    "url": url,
                    "error": str(e)
                })
        
        return self.results
    
    def save_results(self, output_file: str):
        """Save results to JSON file."""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {output_file}")


def demonstrate_custom_chunking():
    """Demonstrate different chunking strategies."""
    print("Chunking Strategy Comparison")
    print("=" * 40)
    
    video_url = "https://www.youtube.com/watch?v=example"
    
    # LangChain chunking
    langchain_config = AnalyzerConfig(
        chunking_method="langchain",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # spaCy chunking
    spacy_config = AnalyzerConfig(
        chunking_method="spacy",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    try:
        print("\n1. LangChain Chunking:")
        langchain_analyzer = YouTubeAnalyzer(langchain_config)
        langchain_result = langchain_analyzer.analyze_video(video_url)
        print(f"   Chunks created: {langchain_result.metadata.chunk_count}")
        
        print("\n2. spaCy Chunking:")
        spacy_analyzer = YouTubeAnalyzer(spacy_config)
        spacy_result = spacy_analyzer.analyze_video(video_url)
        print(f"   Chunks created: {spacy_result.metadata.chunk_count}")
        
        # Compare results
        print("\n3. Comparison:")
        print(f"   LangChain summary length: {len(langchain_result.summary)}")
        print(f"   spaCy summary length: {len(spacy_result.summary)}")
        
    except Exception as e:
        print(f"Error in chunking comparison: {e}")


def demonstrate_multi_provider_analysis():
    """Demonstrate analysis with different LLM providers."""
    print("Multi-Provider Analysis")
    print("=" * 30)
    
    video_url = "https://www.youtube.com/watch?v=example"
    
    providers = [
        ("groq", "Groq"),
        ("openai", "OpenAI"),
        ("local", "Local Model")
    ]
    
    for provider_id, provider_name in providers:
        try:
            print(f"\n{provider_name} Analysis:")
            print("-" * 20)
            
            config = AnalyzerConfig(llm_provider=provider_id)
            analyzer = YouTubeAnalyzer(config)
            
            result = analyzer.analyze_video(video_url)
            
            print(f"Summary: {result.summary[:200]}...")
            print(f"Key points: {len(result.key_points)}")
            
        except Exception as e:
            print(f"Failed with {provider_name}: {e}")


def demonstrate_focused_analysis():
    """Demonstrate focused analysis with specific queries."""
    print("Focused Analysis Examples")
    print("=" * 30)
    
    video_url = "https://www.youtube.com/watch?v=example"
    
    queries = [
        "technical concepts and implementation details",
        "business implications and market impact",
        "future trends and predictions",
        "key statistics and data points"
    ]
    
    try:
        analyzer = YouTubeAnalyzer()
        
        for query in queries:
            print(f"\nQuery: {query}")
            print("-" * 40)
            
            result = analyzer.analyze_video(
                video_url,
                query=query,
                max_chunks=5
            )
            
            print(f"Focused summary: {result.summary[:150]}...")
            
    except Exception as e:
        print(f"Error in focused analysis: {e}")


def demonstrate_transcript_only_analysis():
    """Demonstrate working with transcript text only."""
    print("Transcript-Only Analysis")
    print("=" * 30)
    
    video_url = "https://www.youtube.com/watch?v=example"
    
    try:
        analyzer = YouTubeAnalyzer()
        
        # Get transcript
        transcript = analyzer.get_transcript(video_url)
        print(f"Transcript length: {len(transcript)} characters")
        print(f"First 200 characters: {transcript[:200]}...")
        
        # Manual chunking and embedding
        chunks = [transcript[i:i+1000] for i in range(0, len(transcript), 800)]
        analyzer.create_embeddings(chunks)
        
        # Search within the transcript
        similar_chunks = analyzer.search_similar_content(
            "main topics important points",
            max_results=3
        )
        
        print(f"\nFound {len(similar_chunks)} relevant chunks:")
        for i, chunk in enumerate(similar_chunks, 1):
            print(f"{i}. {chunk.text[:100]}...")
        
    except Exception as e:
        print(f"Error in transcript analysis: {e}")


def create_analysis_report(results: List[Dict[str, Any]], output_file: str):
    """Create a comprehensive analysis report."""
    report = {
        "report_generated": "2024-01-01T00:00:00",
        "total_videos": len(results),
        "successful_analyses": len([r for r in results if "error" not in r]),
        "failed_analyses": len([r for r in results if "error" in r]),
        "summary_statistics": {
            "average_duration": 0,
            "total_chunks": 0,
            "common_themes": []
        },
        "detailed_results": results
    }
    
    # Calculate statistics
    successful_results = [r for r in results if "error" not in r]
    if successful_results:
        durations = [r.get("duration", 0) for r in successful_results if r.get("duration")]
        if durations:
            report["summary_statistics"]["average_duration"] = sum(durations) / len(durations)
        
        total_chunks = sum(r.get("chunk_count", 0) for r in successful_results)
        report["summary_statistics"]["total_chunks"] = total_chunks
    
    # Save report
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Analysis report saved to {output_file}")


def main():
    """Run advanced usage examples."""
    print("YouTube Video Analyzer - Advanced Examples")
    print("=" * 50)
    
    # Example video URLs (replace with actual videos)
    example_videos = [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch?v=example2",
        "https://www.youtube.com/watch?v=example3"
    ]
    
    try:
        # Batch analysis
        print("\n1. Batch Analysis:")
        config = AnalyzerConfig(max_chunks_for_context=3)
        batch_analyzer = BatchAnalyzer(config)
        
        # Process first video only for demo
        results = batch_analyzer.analyze_playlist(example_videos[:1])
        batch_analyzer.save_results("batch_analysis_results.json")
        
        # Create comprehensive report
        create_analysis_report(results, "analysis_report.json")
        
        # Other demonstrations
        print("\n" + "=" * 50)
        demonstrate_custom_chunking()
        
        print("\n" + "=" * 50)
        demonstrate_focused_analysis()
        
        print("\n" + "=" * 50)
        demonstrate_transcript_only_analysis()
        
    except Exception as e:
        print(f"Error in advanced examples: {e}")


if __name__ == "__main__":
    main()