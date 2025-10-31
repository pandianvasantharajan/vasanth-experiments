"""
Embedding generation using Hugging Face sentence-transformers.
"""

import logging
import numpy as np
from typing import List, Optional

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from ..models.schemas import EmbeddingResult

logger = logging.getLogger(__name__)


class Embedder:
    """Generate embeddings using Hugging Face sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
        """
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Dimension: {self.get_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
    
    def get_dimension(self) -> int:
        """
        Get the embedding dimension of the model.
        
        Returns:
            Embedding dimension
        """
        if self.model is None:
            return 0
        return self.model.get_sentence_embedding_dimension()
    
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            EmbeddingResult containing embeddings and metadata
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if not texts:
            return EmbeddingResult(
                embeddings=[],
                chunk_texts=[],
                model_name=self.model_name,
                dimension=self.get_dimension()
            )
        
        try:
            logger.info(f"Encoding {len(texts)} texts with batch size {batch_size}")
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            # Convert to list format for JSON serialization
            embeddings_list = embeddings.tolist()
            
            return EmbeddingResult(
                embeddings=embeddings_list,
                chunk_texts=texts,
                model_name=self.model_name,
                dimension=self.get_dimension()
            )
            
        except Exception as e:
            logger.error(f"Error encoding texts: {e}")
            raise
    
    def encode_single(self, text: str) -> List[float]:
        """
        Encode a single text into an embedding.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector as list of floats
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Error encoding single text: {e}")
            raise
    
    def compute_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity score
        """
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0
    
    def find_most_similar(
        self, 
        query_embedding: List[float], 
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Find the most similar embeddings to a query.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score) tuples
        """
        if not candidate_embeddings:
            return []
        
        try:
            similarities = []
            for i, candidate in enumerate(candidate_embeddings):
                similarity = self.compute_similarity(query_embedding, candidate)
                similarities.append((i, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding similar embeddings: {e}")
            return []