"""
Basic usage example for YouTube Video Analyzer.
"""

import os
from youtube_analyzer import YouTubeAnalyzer, AnalyzerConfig


def main():
    """Demonstrate basic usage of YouTube Video Analyzer."""
    
    # Example YouTube video URL (replace with actual video)
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    
    print("YouTube Video Analyzer - Basic Example")
    print("=" * 50)
    
    try:
        # Method 1: Use default configuration (loads from environment)
        print("\n1. Using default configuration...")
        analyzer = YouTubeAnalyzer()
        
        # Analyze the video
        result = analyzer.analyze_video(video_url)
        
        print(f"Video ID: {result.metadata.video_id}")
        print(f"Analysis completed at: {result.analysis_timestamp}")
        print(f"\nSummary:\n{result.summary}")
        
        if result.key_points:
            print(f"\nKey Points:")
            for i, point in enumerate(result.key_points, 1):
                print(f"{i}. {point}")
        
        print(f"\nGenerated {result.metadata.chunk_count} chunks from transcript")
        
        # Method 2: Use custom configuration
        print("\n" + "=" * 50)
        print("2. Using custom configuration...")
        
        config = AnalyzerConfig(
            llm_provider="groq",  # or "openai", "local", etc.
            chunking_method="spacy",
            chunk_size=800,
            chunk_overlap=100,
            max_chunks_for_context=5
        )
        
        custom_analyzer = YouTubeAnalyzer(config)
        
        # Analyze with a specific query
        focused_result = custom_analyzer.analyze_video(
            video_url,
            query="main topics and conclusions",
            max_chunks=3
        )
        
        print(f"Focused analysis summary:\n{focused_result.summary}")
        
        # Method 3: Search for similar content
        print("\n" + "=" * 50)
        print("3. Searching for similar content...")
        
        similar_chunks = analyzer.search_similar_content(
            query="important information main points",
            max_results=3
        )
        
        print(f"Found {len(similar_chunks)} similar chunks:")
        for i, chunk in enumerate(similar_chunks, 1):
            print(f"\n{i}. Similarity: {chunk.similarity_score:.3f}")
            print(f"   Text: {chunk.text[:100]}...")
        
        # Method 4: Get statistics
        print("\n" + "=" * 50)
        print("4. Database statistics...")
        
        stats = analyzer.get_stats()
        print(f"Total chunks in database: {stats.get('total_chunks', 0)}")
        print(f"Unique videos analyzed: {stats.get('unique_videos', 0)}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have set up your API keys in .env file")
        print("2. Check that the video URL is valid and has transcripts")
        print("3. Ensure all dependencies are installed")


if __name__ == "__main__":
    main()