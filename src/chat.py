"""
Chat Orchestration Module – Medical Chatbot
Coordinates the retrieval-augmented generation (RAG) pipeline:
1. Receive user query
2. Generate query embedding
3. Retrieve relevant passages from vector store
4. Generate context-aware response via LLM
5. Return response with source information
"""

import logging
import time
from typing import Dict, List, Optional

from src.embeddings import generate_single_embedding
from src.llm import generate_response, get_model_info

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """
    Orchestrates the medical chatbot RAG pipeline.
    
    Manages conversation flow, retrieval, and response generation.
    """
    
    def __init__(self, vector_store, top_k: int = 5):
        """
        Initialize the chat orchestrator.
        
        Args:
            vector_store: A VectorStore instance for passage retrieval.
            top_k: Number of passages to retrieve per query.
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.conversation_history: List[Dict] = []
        logger.info(f"ChatOrchestrator initialized (top_k={top_k})")
    
    def process_message(self, user_message: str) -> Dict:
        """
        Process a user message through the full RAG pipeline.
        
        Args:
            user_message: The user's input message.
        
        Returns:
            Dictionary with response, sources, and metadata.
        """
        start_time = time.time()
        
        # Validate input
        if not user_message or not user_message.strip():
            return {
                "response": "Please enter a medical question.",
                "sources": [],
                "processing_time": 0
            }
        
        user_message = user_message.strip()
        
        try:
            # Step 1: Generate query embedding
            query_embedding = generate_single_embedding(user_message)
            
            # Step 2: Retrieve relevant passages
            retrieved = self.vector_store.query(
                query_embedding=query_embedding,
                top_k=self.top_k
            )
            
            # Step 3: Extract passage texts for LLM context
            context_passages = [r["text"] for r in retrieved if r.get("text")]
            
            # Step 4: Generate response via LLM
            response_text = generate_response(
                query=user_message,
                context_passages=context_passages
            )
            
            # Step 5: Build sources list
            sources = []
            for r in retrieved[:3]:  # Top 3 sources
                sources.append({
                    "text": r["text"][:150] + ("..." if len(r["text"]) > 150 else ""),
                    "score": round(r.get("score", 0), 4),
                    "id": r.get("id", "")
                })
            
            processing_time = round(time.time() - start_time, 2)
            
            # Store in conversation history
            entry = {
                "user": user_message,
                "assistant": response_text,
                "sources": sources,
                "timestamp": time.time(),
                "processing_time": processing_time
            }
            self.conversation_history.append(entry)
            
            return {
                "response": response_text,
                "sources": sources,
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            processing_time = round(time.time() - start_time, 2)
            return {
                "response": (
                    "I encountered an error processing your question. "
                    "Please try again or rephrase your question."
                ),
                "sources": [],
                "processing_time": processing_time,
                "error": str(e)
            }
    
    def get_history(self) -> List[Dict]:
        """
        Return the conversation history.
        
        Returns:
            List of conversation entries.
        """
        return [
            {
                "user": entry["user"],
                "assistant": entry["assistant"],
                "timestamp": entry["timestamp"]
            }
            for entry in self.conversation_history
        ]
    
    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")
    
    def get_status(self) -> Dict:
        """
        Get the status of the chatbot system.
        
        Returns:
            Dictionary with system status information.
        """
        return {
            "vector_store_count": self.vector_store.get_count(),
            "conversation_length": len(self.conversation_history),
            "model_info": get_model_info(),
            "top_k": self.top_k
        }
