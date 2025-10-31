"""
LLM-based summarization with support for multiple providers.
"""

import logging
import os
from typing import List, Optional, Dict, Any, Literal
from abc import ABC, abstractmethod

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    import openai
except ImportError:
    openai = None

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    pipeline = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    torch = None

from ..models.schemas import ChunkData

logger = logging.getLogger(__name__)


class BaseSummarizer(ABC):
    """Abstract base class for text summarizers."""
    
    @abstractmethod
    def summarize(self, text: str, context_chunks: Optional[List[ChunkData]] = None) -> str:
        """Generate a summary of the given text."""
        pass
    
    @abstractmethod
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points from the text."""
        pass


class GroqSummarizer(BaseSummarizer):
    """Summarizer using Groq API."""
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        """
        Initialize Groq summarizer.
        
        Args:
            api_key: Groq API key
            model: Model name to use
        """
        if Groq is None:
            raise ImportError("groq is required. Install with: pip install groq")
        
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def summarize(self, text: str, context_chunks: Optional[List[ChunkData]] = None) -> str:
        """
        Generate a summary using Groq.
        
        Args:
            text: Text to summarize (usually the full transcript)
            context_chunks: Optional relevant chunks for context
            
        Returns:
            Generated summary
        """
        try:
            # Prepare context if chunks are provided
            context_text = ""
            if context_chunks:
                context_text = "\n".join([
                    f"[{chunk.start_time:.1f}s-{chunk.end_time:.1f}s]: {chunk.text}"
                    for chunk in context_chunks
                    if chunk.start_time is not None and chunk.end_time is not None
                ])
            
            # Create prompt
            if context_text:
                prompt = f"""Please provide a comprehensive summary of this YouTube video transcript. Focus on the most relevant sections highlighted below:

RELEVANT SECTIONS:
{context_text}

FULL TRANSCRIPT:
{text[:3000]}...  # Truncate for API limits

Please provide:
1. A concise summary (2-3 paragraphs)
2. Main topics covered
3. Key insights or conclusions"""
            else:
                prompt = f"""Please provide a comprehensive summary of this YouTube video transcript:

{text[:4000]}...  # Truncate for API limits

Please provide:
1. A concise summary (2-3 paragraphs)
2. Main topics covered
3. Key insights or conclusions"""
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing video content. Provide clear, concise, and informative summaries."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating summary with Groq: {e}")
            raise
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points using Groq."""
        try:
            prompt = f"""Extract the main key points from this YouTube video transcript. Return them as a numbered list:

{text[:3000]}...

Please provide 5-8 key points that capture the main ideas and important information."""
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key information from video content."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            
            # Parse numbered list
            key_points = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove numbering and bullet points
                    point = line.split('.', 1)[-1].strip()
                    if point:
                        key_points.append(point)
            
            return key_points
            
        except Exception as e:
            logger.error(f"Error extracting key points with Groq: {e}")
            return []


class OpenAISummarizer(BaseSummarizer):
    """Summarizer using OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        """
        Initialize OpenAI summarizer.
        
        Args:
            api_key: OpenAI API key
            model: Model name to use
        """
        if openai is None:
            raise ImportError("openai is required. Install with: pip install openai")
        
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
    
    def summarize(self, text: str, context_chunks: Optional[List[ChunkData]] = None) -> str:
        """Generate a summary using OpenAI."""
        try:
            # Similar implementation to Groq but using OpenAI client
            context_text = ""
            if context_chunks:
                context_text = "\n".join([
                    f"[{chunk.start_time:.1f}s-{chunk.end_time:.1f}s]: {chunk.text}"
                    for chunk in context_chunks
                    if chunk.start_time is not None and chunk.end_time is not None
                ])
            
            if context_text:
                prompt = f"""Please provide a comprehensive summary of this YouTube video transcript. Focus on the most relevant sections highlighted below:

RELEVANT SECTIONS:
{context_text}

FULL TRANSCRIPT:
{text[:3000]}...

Please provide:
1. A concise summary (2-3 paragraphs)
2. Main topics covered
3. Key insights or conclusions"""
            else:
                prompt = f"""Please provide a comprehensive summary of this YouTube video transcript:

{text[:4000]}...

Please provide:
1. A concise summary (2-3 paragraphs)
2. Main topics covered
3. Key insights or conclusions"""
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing video content. Provide clear, concise, and informative summaries."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating summary with OpenAI: {e}")
            raise
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points using OpenAI."""
        try:
            prompt = f"""Extract the main key points from this YouTube video transcript. Return them as a numbered list:

{text[:3000]}...

Please provide 5-8 key points that capture the main ideas and important information."""
            
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert at extracting key information from video content."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model,
                max_tokens=500,
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            
            # Parse numbered list
            key_points = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    point = line.split('.', 1)[-1].strip()
                    if point:
                        key_points.append(point)
            
            return key_points
            
        except Exception as e:
            logger.error(f"Error extracting key points with OpenAI: {e}")
            return []


class LocalSummarizer(BaseSummarizer):
    """Summarizer using local Hugging Face models."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """
        Initialize local summarizer.
        
        Args:
            model_name: Name of the Hugging Face model
        """
        if pipeline is None:
            raise ImportError(
                "transformers is required for local models. "
                "Install with: pip install transformers torch"
            )
        
        self.model_name = model_name
        self.summarizer = None
        self.device = 0 if torch and torch.cuda.is_available() else -1
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the local model."""
        try:
            logger.info(f"Loading local model: {self.model_name}")
            self.summarizer = pipeline(
                "text-generation",
                model=self.model_name,
                device=self.device,
                torch_dtype=torch.float16 if torch and torch.cuda.is_available() else torch.float32
            )
            logger.info("Local model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            raise
    
    def summarize(self, text: str, context_chunks: Optional[List[ChunkData]] = None) -> str:
        """Generate a summary using local model."""
        try:
            # Truncate text for local model limits
            max_length = 1000
            truncated_text = text[:max_length]
            
            prompt = f"Summarize the following video transcript:\n{truncated_text}\n\nSummary:"
            
            response = self.summarizer(
                prompt,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.3,
                pad_token_id=self.summarizer.tokenizer.eos_token_id
            )
            
            generated_text = response[0]['generated_text']
            
            # Extract just the summary part
            summary = generated_text.split("Summary:")[-1].strip()
            
            return summary if summary else "Unable to generate summary"
            
        except Exception as e:
            logger.error(f"Error generating summary with local model: {e}")
            return "Error generating summary"
    
    def extract_key_points(self, text: str) -> List[str]:
        """Extract key points using local model."""
        try:
            # Simple extraction for local models
            sentences = text.split('.')
            # Take the longest sentences as key points
            key_sentences = sorted(sentences, key=len, reverse=True)[:5]
            return [sentence.strip() + "." for sentence in key_sentences if sentence.strip()]
            
        except Exception as e:
            logger.error(f"Error extracting key points with local model: {e}")
            return []


class SummarizerFactory:
    """Factory for creating different types of summarizers."""
    
    @staticmethod
    def create_summarizer(
        provider: Literal["groq", "openai", "openrouter", "local"],
        **kwargs
    ) -> BaseSummarizer:
        """
        Create a summarizer based on the provider.
        
        Args:
            provider: LLM provider
            **kwargs: Additional arguments for the summarizer
            
        Returns:
            Summarizer instance
        """
        if provider == "groq":
            api_key = kwargs.get("groq_api_key") or os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Groq API key is required")
            return GroqSummarizer(api_key, kwargs.get("model", "llama3-8b-8192"))
        
        elif provider == "openai":
            api_key = kwargs.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAISummarizer(api_key, kwargs.get("model", "gpt-3.5-turbo"))
        
        elif provider == "openrouter":
            api_key = kwargs.get("openrouter_api_key") or os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise ValueError("OpenRouter API key is required")
            # Use OpenAI client with OpenRouter base URL
            client = openai.OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            summarizer = OpenAISummarizer.__new__(OpenAISummarizer)
            summarizer.client = client
            summarizer.model = kwargs.get("model", "meta-llama/llama-3.1-8b-instruct:free")
            return summarizer
        
        elif provider == "local":
            model_name = kwargs.get("local_model_name", "microsoft/DialoGPT-medium")
            return LocalSummarizer(model_name)
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")