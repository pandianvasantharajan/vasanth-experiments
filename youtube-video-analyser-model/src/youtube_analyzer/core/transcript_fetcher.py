"""
YouTube transcript fetcher with robust error handling.
"""

import re
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, parse_qs

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api.formatters import TextFormatter
except ImportError:
    YouTubeTranscriptApi = None
    TextFormatter = None

from ..models.schemas import TranscriptSegment, VideoMetadata

logger = logging.getLogger(__name__)


class TranscriptFetcher:
    """Robust YouTube transcript fetcher with fallback mechanisms."""
    
    def __init__(self):
        """Initialize the transcript fetcher."""
        if YouTubeTranscriptApi is None:
            raise ImportError(
                "youtube-transcript-api is required. Install with: "
                "pip install youtube-transcript-api"
            )
        
        self.formatter = TextFormatter()
    
    def extract_video_id(self, url: str) -> str:
        """
        Extract video ID from various YouTube URL formats.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID string
            
        Raises:
            ValueError: If URL format is invalid
        """
        # Handle various YouTube URL formats
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If it's already just an ID
        if re.match(r'^[a-zA-Z0-9_-]{11}$', url):
            return url
            
        raise ValueError(f"Invalid YouTube URL format: {url}")
    
    def get_available_transcripts(self, video_id: str) -> Dict[str, Any]:
        """
        Get available transcript languages for a video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary of available transcripts
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            available = {}
            
            for transcript in transcript_list:
                available[transcript.language_code] = {
                    'language': transcript.language,
                    'is_generated': transcript.is_generated,
                    'is_translatable': transcript.is_translatable
                }
            
            return available
        except Exception as e:
            logger.error(f"Error getting available transcripts: {e}")
            return {}
    
    def fetch_transcript_segments(
        self, 
        video_id: str, 
        language_codes: Optional[List[str]] = None
    ) -> List[TranscriptSegment]:
        """
        Fetch transcript segments with timing information.
        
        Args:
            video_id: YouTube video ID
            language_codes: Preferred language codes (e.g., ['en', 'en-US'])
            
        Returns:
            List of transcript segments
            
        Raises:
            Exception: If transcript cannot be fetched
        """
        try:
            # Try to get transcript in preferred languages
            if language_codes:
                transcript_data = None
                for lang_code in language_codes:
                    try:
                        transcript_data = YouTubeTranscriptApi.get_transcript(
                            video_id, languages=[lang_code]
                        )
                        break
                    except Exception:
                        continue
                
                if transcript_data is None:
                    # Fallback to any available transcript
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            else:
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
            
            # Convert to TranscriptSegment objects
            segments = []
            for segment in transcript_data:
                segments.append(TranscriptSegment(
                    text=segment['text'].strip(),
                    start=segment['start'],
                    duration=segment['duration']
                ))
            
            return segments
            
        except Exception as e:
            logger.error(f"Error fetching transcript for video {video_id}: {e}")
            raise
    
    def fetch_transcript_text(
        self, 
        video_id: str, 
        language_codes: Optional[List[str]] = None
    ) -> str:
        """
        Fetch transcript as a single text string.
        
        Args:
            video_id: YouTube video ID
            language_codes: Preferred language codes
            
        Returns:
            Complete transcript text
        """
        segments = self.fetch_transcript_segments(video_id, language_codes)
        return ' '.join([segment.text for segment in segments])
    
    def get_video_metadata(
        self, 
        video_id: str, 
        transcript_segments: Optional[List[TranscriptSegment]] = None
    ) -> VideoMetadata:
        """
        Extract metadata from video and transcript.
        
        Args:
            video_id: YouTube video ID
            transcript_segments: Optional transcript segments
            
        Returns:
            Video metadata
        """
        metadata = VideoMetadata(video_id=video_id)
        
        if transcript_segments:
            # Calculate duration from transcript
            if transcript_segments:
                last_segment = transcript_segments[-1]
                metadata.duration = last_segment.end
            
            # Calculate transcript length
            full_text = ' '.join([seg.text for seg in transcript_segments])
            metadata.transcript_length = len(full_text)
        
        # Try to get additional metadata from available transcripts
        try:
            available_transcripts = self.get_available_transcripts(video_id)
            if available_transcripts:
                # Use the first available language as detected language
                first_lang = list(available_transcripts.keys())[0]
                metadata.language = first_lang
        except Exception:
            pass
        
        return metadata
    
    def fetch_video_data(
        self, 
        url: str, 
        language_codes: Optional[List[str]] = None
    ) -> tuple[List[TranscriptSegment], VideoMetadata]:
        """
        Fetch complete video data including transcript and metadata.
        
        Args:
            url: YouTube URL or video ID
            language_codes: Preferred language codes
            
        Returns:
            Tuple of (transcript_segments, metadata)
        """
        video_id = self.extract_video_id(url)
        segments = self.fetch_transcript_segments(video_id, language_codes)
        metadata = self.get_video_metadata(video_id, segments)
        
        return segments, metadata