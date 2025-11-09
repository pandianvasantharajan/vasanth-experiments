"""
Vector retrieval using ChromaDB for efficient similarity search.
"""

import logging
import time
import os
from typing import List, Optional, Dict, Any

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
except ImportError:
    chromadb = None
    ChromaSettings = None

from ..models.schemas import ChunkData, RetrievalResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Vector retrieval system using ChromaDB."""
    
    def __init__(
        self, 
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "youtube_transcripts"
    ):
        """
        Initialize the vector retriever.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        if chromadb is None:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Ensure persist directory exists
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_chunks(
        self, 
        chunks: List[ChunkData], 
        embeddings: List[List[float]],
        video_id: str
    ) -> None:
        """
        Add chunks and their embeddings to the vector database.
        
        Args:
            chunks: List of chunk data
            embeddings: Corresponding embeddings
            video_id: YouTube video ID for metadata
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        try:
            # Prepare data for ChromaDB
            documents = [chunk.text for chunk in chunks]
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{video_id}_{chunk.chunk_index}_{i}"
                ids.append(chunk_id)
                
                # ChromaDB doesn't accept None values, so filter them out
                metadata = {
                    "video_id": video_id,
                    "chunk_index": int(chunk.chunk_index) if chunk.chunk_index is not None else 0,
                    "start_time": float(chunk.start_time) if chunk.start_time is not None else 0.0,
                    "end_time": float(chunk.end_time) if chunk.end_time is not None else 0.0,
                    "text_length": len(chunk.text)
                }
                # Remove any remaining None values
                metadata = {k: v for k, v in metadata.items() if v is not None}
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(chunks)} chunks for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error adding chunks to database: {e}")
            raise
    
    def search_similar_chunks(
        self, 
        query_embedding: List[float], 
        n_results: int = 10,
        video_id: Optional[str] = None
    ) -> RetrievalResult:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            video_id: Optional video ID to filter results
            
        Returns:
            RetrievalResult with similar chunks
        """
        start_time = time.time()
        
        try:
            # Prepare where clause for filtering
            where_clause = None
            if video_id:
                where_clause = {"video_id": video_id}
            
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert results to ChunkData objects
            chunks = []
            documents = results["documents"][0] if results["documents"] else []
            metadatas = results["metadatas"][0] if results["metadatas"] else []
            distances = results["distances"][0] if results["distances"] else []
            
            for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                # Convert distance to similarity score (1 - distance for cosine)
                similarity_score = 1.0 - distance if distance is not None else 0.0
                
                chunk = ChunkData(
                    text=doc,
                    start_time=metadata.get("start_time"),
                    end_time=metadata.get("end_time"),
                    chunk_index=metadata.get("chunk_index", i),
                    similarity_score=similarity_score
                )
                chunks.append(chunk)
            
            retrieval_time = time.time() - start_time
            
            # Get total count for metadata
            total_count = self.collection.count()
            
            return RetrievalResult(
                chunks=chunks,
                query="",  # Query text not stored with embedding
                total_chunks_available=total_count,
                retrieval_time=retrieval_time
            )
            
        except Exception as e:
            logger.error(f"Error searching for similar chunks: {e}")
            raise
    
    def get_chunks_by_video(self, video_id: str) -> List[ChunkData]:
        """
        Get all chunks for a specific video.
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            List of chunks for the video
        """
        try:
            results = self.collection.get(
                where={"video_id": video_id},
                include=["documents", "metadatas"]
            )
            
            chunks = []
            documents = results["documents"] if results["documents"] else []
            metadatas = results["metadatas"] if results["metadatas"] else []
            
            for doc, metadata in zip(documents, metadatas):
                chunk = ChunkData(
                    text=doc,
                    start_time=metadata.get("start_time"),
                    end_time=metadata.get("end_time"),
                    chunk_index=metadata.get("chunk_index", 0)
                )
                chunks.append(chunk)
            
            # Sort by chunk index
            chunks.sort(key=lambda x: x.chunk_index)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting chunks for video {video_id}: {e}")
            return []
    
    def delete_video_chunks(self, video_id: str) -> None:
        """
        Delete all chunks for a specific video.
        
        Args:
            video_id: YouTube video ID
        """
        try:
            # Get all chunk IDs for the video
            results = self.collection.get(
                where={"video_id": video_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for video {video_id}")
            
        except Exception as e:
            logger.error(f"Error deleting chunks for video {video_id}: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get unique video IDs
            results = self.collection.get(include=["metadatas"])
            video_ids = set()
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    if "video_id" in metadata:
                        video_ids.add(metadata["video_id"])
            
            return {
                "total_chunks": count,
                "unique_videos": len(video_ids),
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def clear_collection(self) -> None:
        """Clear all data from the collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(name=self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise