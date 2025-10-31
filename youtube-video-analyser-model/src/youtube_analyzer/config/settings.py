"""
Settings management using Pydantic Settings.
"""

import os
from typing import Optional, Literal
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # LLM Configuration
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openrouter_api_key: Optional[str] = Field(default=None, alias="OPENROUTER_API_KEY")
    
    # Model Configuration
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    llm_provider: Literal["groq", "openai", "openrouter", "local"] = Field(
        default="groq", alias="LLM_PROVIDER"
    )
    local_model_name: str = Field(
        default="microsoft/DialoGPT-medium", alias="LOCAL_MODEL_NAME"
    )
    
    # Chunking Configuration
    chunking_method: Literal["langchain", "spacy"] = Field(
        default="langchain", alias="CHUNKING_METHOD"
    )
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    max_chunks_for_context: int = Field(default=10, alias="MAX_CHUNKS_FOR_CONTEXT")
    
    # ChromaDB Configuration
    chroma_persist_directory: str = Field(
        default="./data/chroma_db", alias="CHROMA_PERSIST_DIRECTORY"
    )
    collection_name: str = Field(
        default="youtube_transcripts", alias="COLLECTION_NAME"
    )
    
    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_file: str = Field(default="./data/youtube_analyzer.log", alias="LOG_FILE")
    
    # Performance Configuration
    batch_size: int = Field(default=32, alias="BATCH_SIZE")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")
    request_timeout: int = Field(default=30, alias="REQUEST_TIMEOUT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()