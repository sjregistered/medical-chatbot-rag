"""
Embeddings Module – Medical Chatbot
Generates sentence embeddings using sentence-transformers.
Uses the lightweight all-MiniLM-L6-v2 model for fast, effective embeddings.
"""

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Global model instance (lazy-loaded)
_model = None
_model_name = "all-MiniLM-L6-v2"


def get_model():
    """
    Lazy-load the sentence-transformer model.
    Keeps a single global instance to avoid repeated loading.
    
    Returns:
        SentenceTransformer model instance.
    """
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {_model_name}")
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
        logger.info("Embedding model loaded successfully.")
    return _model


def generate_embeddings(texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """
    Generate embeddings for a list of text strings.
    
    Args:
        texts: List of text strings to embed.
        batch_size: Number of texts to process at once.
    
    Returns:
        List of embedding vectors (each a list of floats).
    """
    if not texts:
        return []
    
    model = get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    
    return embeddings.tolist()


def generate_single_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Text string to embed.
    
    Returns:
        Embedding vector as a list of floats.
    """
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.
    
    Returns:
        Cosine similarity score between -1 and 1.
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return float(dot_product / (norm_a * norm_b))


def get_embedding_dimension() -> int:
    """
    Return the dimensionality of the embedding model's output.
    For all-MiniLM-L6-v2, this is 384.
    
    Returns:
        Integer dimension of embedding vectors.
    """
    model = get_model()
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()
