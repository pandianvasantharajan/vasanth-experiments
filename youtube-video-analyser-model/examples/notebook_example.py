"""
Jupyter Notebook example for YouTube Video Analyzer.

This notebook demonstrates interactive usage of the analyzer.
"""

# Cell 1: Setup and imports
import os
import json
from pathlib import Path

# Add src to path for development
import sys
sys.path.append('../src')

from youtube_analyzer import YouTubeAnalyzer, AnalyzerConfig
from youtube_analyzer.utils import format_duration, validate_youtube_url

print("YouTube Video Analyzer - Jupyter Notebook Example")
print("=" * 50)

# Cell 2: Configuration
# Set up your API keys (create a .env file in project root)
config = AnalyzerConfig(
    llm_provider="groq",  # Change to "openai", "local", etc.
    chunking_method="langchain",
    chunk_size=1000,
    chunk_overlap=200,
    max_chunks_for_context=5
)

analyzer = YouTubeAnalyzer(config)
print("✓ Analyzer initialized successfully")

# Cell 3: Basic video analysis
video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with actual video

# Validate URL first
if validate_youtube_url(video_url):
    print("✓ Valid YouTube URL")
    
    try:
        result = analyzer.analyze_video(video_url)
        
        print(f"\n📺 Video Analysis Results")
        print(f"Video ID: {result.metadata.video_id}")
        print(f"Duration: {format_duration(result.metadata.duration) if result.metadata.duration else 'Unknown'}")
        print(f"Chunks created: {result.metadata.chunk_count}")
        print(f"Analysis completed: {result.analysis_timestamp}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
else:
    print("❌ Invalid YouTube URL")

# Cell 4: Display summary
if 'result' in locals():
    print("\n📝 Summary:")
    print("-" * 40)
    print(result.summary)

# Cell 5: Display key points
if 'result' in locals() and result.key_points:
    print("\n🔑 Key Points:")
    print("-" * 40)
    for i, point in enumerate(result.key_points, 1):
        print(f"{i}. {point}")

# Cell 6: Context chunks analysis
if 'result' in locals() and result.context_chunks:
    print("\n🔍 Most Relevant Chunks:")
    print("-" * 40)
    
    for i, chunk in enumerate(result.context_chunks[:3], 1):
        print(f"\n{i}. Similarity: {chunk.similarity_score:.3f}")
        if chunk.start_time and chunk.end_time:
            print(f"   Time: {chunk.start_time:.1f}s - {chunk.end_time:.1f}s")
        print(f"   Text: {chunk.text[:200]}...")

# Cell 7: Focused analysis with custom query
custom_query = "technical concepts and implementation details"

try:
    focused_result = analyzer.analyze_video(
        video_url,
        query=custom_query,
        max_chunks=3
    )
    
    print(f"\n🎯 Focused Analysis (Query: '{custom_query}'):")
    print("-" * 40)
    print(focused_result.summary)
    
except Exception as e:
    print(f"❌ Focused analysis error: {e}")

# Cell 8: Search similar content
search_query = "important main points key information"

try:
    similar_chunks = analyzer.search_similar_content(
        query=search_query,
        max_results=5
    )
    
    print(f"\n🔎 Search Results for '{search_query}':")
    print("-" * 40)
    
    for i, chunk in enumerate(similar_chunks, 1):
        print(f"\n{i}. Similarity: {chunk.similarity_score:.3f}")
        print(f"   {chunk.text[:150]}...")
        
except Exception as e:
    print(f"❌ Search error: {e}")

# Cell 9: Database statistics
try:
    stats = analyzer.get_stats()
    
    print("\n📊 Database Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
        
except Exception as e:
    print(f"❌ Stats error: {e}")

# Cell 10: Export results
if 'result' in locals():
    # Create export data
    export_data = {
        "video_analysis": {
            "url": video_url,
            "video_id": result.metadata.video_id,
            "summary": result.summary,
            "key_points": result.key_points,
            "metadata": {
                "duration": result.metadata.duration,
                "language": result.metadata.language,
                "transcript_length": result.metadata.transcript_length,
                "chunk_count": result.metadata.chunk_count
            },
            "analysis_timestamp": result.analysis_timestamp,
            "config_used": result.config_used
        }
    }
    
    # Save to file
    output_file = "notebook_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n💾 Results exported to: {output_file}")

# Cell 11: Visualization (if matplotlib is available)
try:
    import matplotlib.pyplot as plt
    import numpy as np
    
    if 'result' in locals() and result.context_chunks:
        # Create similarity score visualization
        chunks = result.context_chunks[:10]  # Top 10 chunks
        scores = [chunk.similarity_score for chunk in chunks if chunk.similarity_score]
        
        if scores:
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(scores)), scores)
            plt.title('Chunk Similarity Scores')
            plt.xlabel('Chunk Index')
            plt.ylabel('Similarity Score')
            plt.xticks(range(len(scores)))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        
        # Time distribution of relevant chunks
        times = [(chunk.start_time, chunk.end_time) for chunk in chunks 
                if chunk.start_time and chunk.end_time]
        
        if times:
            start_times = [t[0] for t in times]
            durations = [t[1] - t[0] for t in times]
            
            plt.figure(figsize=(12, 4))
            plt.barh(range(len(start_times)), durations, left=start_times)
            plt.title('Timeline of Most Relevant Chunks')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Chunk Index')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

except ImportError:
    print("\n📊 Install matplotlib for visualizations: pip install matplotlib")

print("\n✅ Notebook example completed!")
print("Try modifying the video URL and queries to explore different content.")