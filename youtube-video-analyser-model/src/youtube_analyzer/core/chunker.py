"""
Text chunking strategies using LangChain and spaCy.
"""

import logging
from typing import List, Optional, Literal
from abc import ABC, abstractmethod

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

try:
    import spacy
    from spacy.lang.en import English
except ImportError:
    spacy = None
    English = None

from ..models.schemas import TranscriptSegment, ChunkData

logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Abstract base class for text chunkers."""
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text into smaller segments."""
        pass
    
    @abstractmethod
    def chunk_segments(self, segments: List[TranscriptSegment]) -> List[ChunkData]:
        """Chunk transcript segments with timing information."""
        pass


class LangChainChunker(BaseChunker):
    """Text chunker using LangChain's RecursiveCharacterTextSplitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize LangChain chunker.
        
        Args:
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between consecutive chunks
        """
        if RecursiveCharacterTextSplitter is None:
            raise ImportError(
                "langchain is required for LangChainChunker. "
                "Install with: pip install langchain"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using LangChain splitter.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)
    
    def chunk_segments(self, segments: List[TranscriptSegment]) -> List[ChunkData]:
        """
        Chunk transcript segments while preserving timing information.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of chunk data with timing
        """
        # Combine segments into text with markers
        full_text = ""
        segment_markers = []  # (position, segment_index)
        
        for i, segment in enumerate(segments):
            start_pos = len(full_text)
            full_text += segment.text + " "
            end_pos = len(full_text) - 1
            segment_markers.append((start_pos, end_pos, i))
        
        # Chunk the full text
        chunks = self.chunk_text(full_text)
        
        # Map chunks back to timing information
        chunk_data = []
        text_position = 0
        
        for chunk_index, chunk in enumerate(chunks):
            chunk_start_pos = text_position
            chunk_end_pos = text_position + len(chunk)
            
            # Find overlapping segments
            overlapping_segments = []
            for start_pos, end_pos, seg_idx in segment_markers:
                if not (end_pos < chunk_start_pos or start_pos > chunk_end_pos):
                    overlapping_segments.append(seg_idx)
            
            # Calculate timing from overlapping segments
            start_time = None
            end_time = None
            if overlapping_segments:
                first_seg = segments[overlapping_segments[0]]
                last_seg = segments[overlapping_segments[-1]]
                start_time = first_seg.start
                end_time = last_seg.end
            
            chunk_data.append(ChunkData(
                text=chunk.strip(),
                start_time=start_time,
                end_time=end_time,
                chunk_index=chunk_index
            ))
            
            text_position = chunk_end_pos + 1  # Account for spaces
        
        return chunk_data


class SpacyChunker(BaseChunker):
    """Text chunker using spaCy for semantic sentence-based chunking."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize spaCy chunker.
        
        Args:
            chunk_size: Target size of each chunk (in characters)
            chunk_overlap: Overlap between consecutive chunks
        """
        if spacy is None:
            raise ImportError(
                "spacy is required for SpacyChunker. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Fallback to basic English tokenizer
            self.nlp = English()
            self.nlp.add_pipe('sentencizer')
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text using spaCy sentence segmentation.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of text chunks
        """
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, start new chunk
            if (len(current_chunk) + len(sentence) > self.chunk_size and 
                current_chunk):
                chunks.append(current_chunk.strip())
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_segments(self, segments: List[TranscriptSegment]) -> List[ChunkData]:
        """
        Chunk transcript segments using semantic boundaries.
        
        Args:
            segments: List of transcript segments
            
        Returns:
            List of chunk data with timing
        """
        # Process segments to find sentence boundaries
        processed_segments = []
        
        for segment in segments:
            doc = self.nlp(segment.text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            
            if sentences:
                # Distribute time across sentences
                time_per_sentence = segment.duration / len(sentences)
                for i, sentence in enumerate(sentences):
                    sent_start = segment.start + (i * time_per_sentence)
                    sent_duration = time_per_sentence
                    
                    processed_segments.append(TranscriptSegment(
                        text=sentence,
                        start=sent_start,
                        duration=sent_duration
                    ))
        
        # Now chunk the processed segments
        chunks = []
        current_chunk_segments = []
        current_length = 0
        
        for segment in processed_segments:
            # Check if adding this segment would exceed chunk size
            if (current_length + len(segment.text) > self.chunk_size and 
                current_chunk_segments):
                
                # Create chunk from current segments
                chunk_text = " ".join([seg.text for seg in current_chunk_segments])
                start_time = current_chunk_segments[0].start
                end_time = current_chunk_segments[-1].end
                
                chunks.append(ChunkData(
                    text=chunk_text,
                    start_time=start_time,
                    end_time=end_time,
                    chunk_index=len(chunks)
                ))
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_segments = []
                    overlap_length = 0
                    
                    # Take segments from the end for overlap
                    for seg in reversed(current_chunk_segments):
                        if overlap_length + len(seg.text) <= self.chunk_overlap:
                            overlap_segments.insert(0, seg)
                            overlap_length += len(seg.text)
                        else:
                            break
                    
                    current_chunk_segments = overlap_segments + [segment]
                    current_length = sum(len(seg.text) for seg in current_chunk_segments)
                else:
                    current_chunk_segments = [segment]
                    current_length = len(segment.text)
            else:
                current_chunk_segments.append(segment)
                current_length += len(segment.text)
        
        # Add the last chunk
        if current_chunk_segments:
            chunk_text = " ".join([seg.text for seg in current_chunk_segments])
            start_time = current_chunk_segments[0].start
            end_time = current_chunk_segments[-1].end
            
            chunks.append(ChunkData(
                text=chunk_text,
                start_time=start_time,
                end_time=end_time,
                chunk_index=len(chunks)
            ))
        
        return chunks


class ChunkerFactory:
    """Factory for creating different types of chunkers."""
    
    @staticmethod
    def create_chunker(
        method: Literal["langchain", "spacy"],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> BaseChunker:
        """
        Create a chunker based on the specified method.
        
        Args:
            method: Chunking method ("langchain" or "spacy")
            chunk_size: Target size of each chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            Chunker instance
            
        Raises:
            ValueError: If method is not supported
        """
        if method == "langchain":
            return LangChainChunker(chunk_size, chunk_overlap)
        elif method == "spacy":
            return SpacyChunker(chunk_size, chunk_overlap)
        else:
            raise ValueError(f"Unsupported chunking method: {method}")