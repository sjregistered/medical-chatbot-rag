"""
Vector Store Module – Medical Chatbot
Provides a unified interface for vector storage and retrieval.
Supports Pinecone (primary) with automatic Chroma fallback.
"""

import os
import logging
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Abstract-ish base via a simple class hierarchy ───

class VectorStore:
    """Base class for vector store implementations."""
    
    def add_documents(self, ids: List[str], texts: List[str],
                      embeddings: List[List[float]],
                      metadatas: Optional[List[Dict]] = None) -> None:
        raise NotImplementedError
    
    def query(self, query_embedding: List[float],
              top_k: int = 5) -> List[Dict]:
        raise NotImplementedError
    
    def get_count(self) -> int:
        raise NotImplementedError
    
    def delete_all(self) -> None:
        raise NotImplementedError


# ─── Chroma Implementation (Local Fallback) ───

class ChromaVectorStore(VectorStore):
    """
    Chroma-based local vector store.
    Used as fallback when Pinecone API key is not available.
    """
    
    def __init__(self, collection_name: str = "medical_chatbot",
                 persist_directory: str = "./chroma_db"):
        import chromadb
        from chromadb.config import Settings
        
        self.persist_directory = persist_directory
        try:
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        except AttributeError:
            # Very old chromadb versions (<0.4) without PersistentClient
            self.client = chromadb.Client(Settings(
                persist_directory=persist_directory,
                anonymized_telemetry=False
            ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(
            f"Chroma vector store initialized: collection='{collection_name}', "
            f"existing_count={self.collection.count()}"
        )
    
    def add_documents(self, ids: List[str], texts: List[str],
                      embeddings: List[List[float]],
                      metadatas: Optional[List[Dict]] = None) -> None:
        """Add documents with pre-computed embeddings to Chroma."""
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas or [{}] * len(ids)
        )
        logger.info(f"Added {len(ids)} documents to Chroma.")
    
    def query(self, query_embedding: List[float],
              top_k: int = 5) -> List[Dict]:
        """Query Chroma for similar documents."""
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"]
        )
        
        formatted = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append({
                    "id": results["ids"][0][i] if results["ids"] else f"doc_{i}",
                    "text": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0
                })
        
        return formatted
    
    def get_count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()
    
    def delete_all(self) -> None:
        """Delete all documents from the collection."""
        # Chroma doesn't have a bulk delete, so we recreate the collection
        name = self.collection.name
        metadata = self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.create_collection(
            name=name, metadata=metadata
        )
        logger.info("Deleted all documents from Chroma collection.")


# ─── Pinecone Implementation (Primary) ───

class PineconeVectorStore(VectorStore):
    """
    Pinecone-based cloud vector store.
    Requires PINECONE_API_KEY environment variable.
    """
    
    def __init__(self, index_name: str = "medical-chatbot",
                 dimension: int = 384):
        from pinecone import Pinecone, ServerlessSpec
        
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not set.")
        
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.dimension = dimension
        
        # Create index if it doesn't exist
        existing = [idx.name for idx in self.pc.list_indexes()]
        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Created Pinecone index: {index_name}")
        
        self.index = self.pc.Index(index_name)
        logger.info(f"Pinecone vector store initialized: index='{index_name}'")
    
    def add_documents(self, ids: List[str], texts: List[str],
                      embeddings: List[List[float]],
                      metadatas: Optional[List[Dict]] = None) -> None:
        """Upsert documents into Pinecone."""
        vectors = []
        for i in range(len(ids)):
            meta = metadatas[i] if metadatas else {}
            meta["text"] = texts[i]  # Store text in metadata
            vectors.append({
                "id": ids[i],
                "values": embeddings[i],
                "metadata": meta
            })
        
        # Upsert in batches of 100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        logger.info(f"Upserted {len(ids)} documents to Pinecone.")
    
    def query(self, query_embedding: List[float],
              top_k: int = 5) -> List[Dict]:
        """Query Pinecone for similar documents."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        formatted = []
        for match in results.get("matches", []):
            text = match.get("metadata", {}).pop("text", "")
            formatted.append({
                "id": match["id"],
                "text": text,
                "metadata": match.get("metadata", {}),
                "score": match.get("score", 0.0)
            })
        
        return formatted
    
    def get_count(self) -> int:
        """Return the approximate number of vectors in the index."""
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)
    
    def delete_all(self) -> None:
        """Delete all vectors in the index."""
        self.index.delete(delete_all=True)
        logger.info("Deleted all vectors from Pinecone index.")


# ─── Factory Function ───

def create_vector_store(
    collection_name: str = "medical_chatbot",
    dimension: int = 384,
    persist_directory: str = "./chroma_db"
) -> VectorStore:
    """
    Create the appropriate vector store based on available configuration.
    
    Tries Pinecone first; falls back to Chroma if no API key is set.
    
    Args:
        collection_name: Name for the collection/index.
        dimension: Embedding dimension (384 for MiniLM).
        persist_directory: Local directory for Chroma persistence.
    
    Returns:
        A VectorStore instance (Pinecone or Chroma).
    """
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    
    if pinecone_key:
        try:
            store = PineconeVectorStore(
                index_name=collection_name.replace("_", "-"),
                dimension=dimension
            )
            logger.info("Using Pinecone vector store.")
            return store
        except Exception as e:
            logger.warning(f"Pinecone initialization failed: {e}. Falling back to Chroma.")
    
    logger.info("Using Chroma vector store (local fallback).")
    return ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
