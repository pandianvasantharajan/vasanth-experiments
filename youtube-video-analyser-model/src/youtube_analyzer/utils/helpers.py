"""
Utility functions and helpers for the YouTube Analyzer.
"""

import re
import logging
import time
from typing import List, Dict, Any, Optional
from functools import wraps
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


def retry_with_exponential_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
            
            return None
        return wrapper
    return decorator


def validate_youtube_url(url: str) -> bool:
    """
    Validate if a URL is a valid YouTube URL.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid YouTube URL, False otherwise
    """
    youtube_patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)',
    ]
    
    for pattern in youtube_patterns:
        if re.search(pattern, url):
            return True
    
    # Check if it's just a video ID
    if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
        return True
    
    return False


def extract_video_metadata_from_url(url: str) -> Dict[str, Any]:
    """
    Extract metadata from YouTube URL parameters.
    
    Args:
        url: YouTube URL
        
    Returns:
        Dictionary with extracted metadata
    """
    metadata = {}
    
    try:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        
        # Extract video ID
        if 'v' in params:
            metadata['video_id'] = params['v'][0]
        
        # Extract playlist info
        if 'list' in params:
            metadata['playlist_id'] = params['list'][0]
        
        # Extract timestamp
        if 't' in params:
            metadata['start_time'] = params['t'][0]
        elif parsed.fragment:
            # Handle #t=123 format
            if parsed.fragment.startswith('t='):
                metadata['start_time'] = parsed.fragment[2:]
    
    except Exception as e:
        logger.warning(f"Error extracting metadata from URL: {e}")
    
    return metadata


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        remaining_seconds = seconds % 60
        return f"{hours}h {minutes}m {remaining_seconds:.1f}s"


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Strip and return
    return text.strip()


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity using Jaccard similarity.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score between 0 and 1
    """
    if not text1 or not text2:
        return 0.0
    
    # Convert to sets of words
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    # Calculate Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def truncate_text(text: str, max_length: int, add_ellipsis: bool = True) -> str:
    """
    Truncate text to specified length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        add_ellipsis: Whether to add ellipsis
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    if add_ellipsis and max_length > 3:
        return text[:max_length-3] + "..."
    else:
        return text[:max_length]


def merge_overlapping_chunks(chunks: List[Dict[str, Any]], overlap_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Merge chunks that have significant overlap.
    
    Args:
        chunks: List of chunk dictionaries with 'text' key
        overlap_threshold: Threshold for considering chunks overlapping
        
    Returns:
        List of merged chunks
    """
    if not chunks:
        return []
    
    merged = []
    current_chunk = chunks[0].copy()
    
    for next_chunk in chunks[1:]:
        similarity = calculate_text_similarity(
            current_chunk['text'], 
            next_chunk['text']
        )
        
        if similarity >= overlap_threshold:
            # Merge chunks
            current_chunk['text'] = current_chunk['text'] + " " + next_chunk['text']
            # Update timing if available
            if 'end_time' in next_chunk:
                current_chunk['end_time'] = next_chunk['end_time']
        else:
            merged.append(current_chunk)
            current_chunk = next_chunk.copy()
    
    # Add the last chunk
    merged.append(current_chunk)
    
    return merged


def estimate_reading_time(text: str, words_per_minute: int = 200) -> float:
    """
    Estimate reading time for text.
    
    Args:
        text: Text to estimate reading time for
        words_per_minute: Average reading speed
        
    Returns:
        Estimated reading time in minutes
    """
    if not text:
        return 0.0
    
    word_count = len(text.split())
    return word_count / words_per_minute


def extract_key_phrases(text: str, max_phrases: int = 10) -> List[str]:
    """
    Extract key phrases from text using simple heuristics.
    
    Args:
        text: Input text
        max_phrases: Maximum number of phrases to extract
        
    Returns:
        List of key phrases
    """
    if not text:
        return []
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', text)
    
    # Score sentences by length and word frequency
    scored_sentences = []
    word_freq = {}
    
    # Calculate word frequency
    words = text.lower().split()
    for word in words:
        word = re.sub(r'[^\w]', '', word)
        if len(word) > 2:  # Ignore very short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Score sentences
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Ignore very short sentences
            words_in_sentence = sentence.lower().split()
            score = sum(word_freq.get(re.sub(r'[^\w]', '', word), 0) 
                       for word in words_in_sentence)
            scored_sentences.append((score, sentence))
    
    # Sort by score and return top phrases
    scored_sentences.sort(reverse=True)
    
    return [sentence for _, sentence in scored_sentences[:max_phrases]]


def create_summary_template() -> str:
    """
    Create a template for summary output.
    
    Returns:
        Summary template string
    """
    return """
## Video Summary

**Main Topic:** [Topic]

**Key Points:**
1. [Point 1]
2. [Point 2]
3. [Point 3]

**Summary:**
[Detailed summary paragraph]

**Takeaways:**
- [Takeaway 1]
- [Takeaway 2]
- [Takeaway 3]
"""


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors
    """
    errors = []
    
    # Required fields
    required_fields = ['llm_provider', 'embedding_model', 'chunking_method']
    for field in required_fields:
        if field not in config or not config[field]:
            errors.append(f"Missing required field: {field}")
    
    # Validate chunk size
    if 'chunk_size' in config:
        if not isinstance(config['chunk_size'], int) or config['chunk_size'] <= 0:
            errors.append("chunk_size must be a positive integer")
    
    # Validate chunk overlap
    if 'chunk_overlap' in config:
        if not isinstance(config['chunk_overlap'], int) or config['chunk_overlap'] < 0:
            errors.append("chunk_overlap must be a non-negative integer")
        
        if 'chunk_size' in config and config['chunk_overlap'] >= config['chunk_size']:
            errors.append("chunk_overlap must be less than chunk_size")
    
    # Validate LLM provider
    valid_providers = ['groq', 'openai', 'openrouter', 'local']
    if config.get('llm_provider') not in valid_providers:
        errors.append(f"llm_provider must be one of: {valid_providers}")
    
    # Validate chunking method
    valid_methods = ['langchain', 'spacy']
    if config.get('chunking_method') not in valid_methods:
        errors.append(f"chunking_method must be one of: {valid_methods}")
    
    return errors