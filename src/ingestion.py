"""
Ingestion Module – Medical Chatbot
Handles loading medical knowledge corpus and chunking text into
manageable pieces for embedding and vector storage.
"""

import os
import re
from typing import List, Dict


def load_medical_corpus(data_dir: str = "data") -> str:
    """
    Load the medical knowledge corpus from text files in the data directory.
    
    Args:
        data_dir: Path to the directory containing medical text files.
    
    Returns:
        Combined text content from all files.
    """
    corpus_text = ""
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found.")
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith((".txt", ".md")):
            filepath = os.path.join(data_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                corpus_text += f.read() + "\n\n"
    
    if not corpus_text.strip():
        raise ValueError("No text content found in data directory.")
    
    return corpus_text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Dict[str, str]]:
    """
    Split text into overlapping chunks for embedding.
    
    Uses sentence-aware splitting: tries to break at sentence boundaries
    to preserve semantic coherence within chunks.
    
    Args:
        text: The full text to chunk.
        chunk_size: Target number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.
    
    Returns:
        List of dicts with 'id', 'text', and 'metadata' keys.
    """
    chunks = []
    chunk_id = 0

    def _emit(chunk_sentences: List[str], section: str) -> None:
        nonlocal chunk_id
        body = " ".join(chunk_sentences).strip()
        if not body:
            return
        chunks.append({
            "id": f"chunk_{chunk_id}",
            "text": body,
            "metadata": {
                "chunk_index": chunk_id,
                "char_count": len(body),
                "section": section,
                "source": "medical_knowledge_corpus"
            }
        })
        chunk_id += 1

    # Split corpus into sections (separated by '---' lines) so chunks
    # never straddle two unrelated topics.
    sections = [s.strip() for s in re.split(r'\n\s*-{3,}\s*\n', text) if s.strip()]

    for section_text in sections:
        lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
        # Use an ALL-CAPS first line as the section title
        title = lines[0] if lines and lines[0].isupper() else ""
        body = " ".join(lines[1:] if title else lines)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', body) if s.strip()]

        current: List[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) > chunk_size:
                _emit(current, title)
                # Sentence-aligned overlap: carry over trailing whole sentences
                overlap: List[str] = []
                overlap_len = 0
                for prev in reversed(current):
                    if overlap_len + len(prev) > chunk_overlap:
                        break
                    overlap.insert(0, prev)
                    overlap_len += len(prev) + 1
                current = overlap
                current_len = overlap_len
            current.append(sentence)
            current_len += len(sentence) + 1

        if current:
            _emit(current, title)

    # Prefix each chunk with its topic so retrieval/LLM keep context
    for c in chunks:
        section = c["metadata"]["section"]
        if section and not c["text"].upper().startswith(section):
            c["text"] = f"{section.title()}: {c['text']}"
            c["metadata"]["char_count"] = len(c["text"])

    return chunks


def get_ingestion_stats(chunks: List[Dict]) -> Dict:
    """
    Return statistics about the ingested and chunked corpus.
    
    Args:
        chunks: List of chunk dictionaries.
    
    Returns:
        Dictionary with ingestion statistics.
    """
    total_chars = sum(c["metadata"]["char_count"] for c in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    
    return {
        "total_chunks": len(chunks),
        "total_characters": total_chars,
        "average_chunk_size": round(avg_chars, 1),
        "min_chunk_size": min(c["metadata"]["char_count"] for c in chunks) if chunks else 0,
        "max_chunk_size": max(c["metadata"]["char_count"] for c in chunks) if chunks else 0,
    }
