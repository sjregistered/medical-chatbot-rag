"""
Chat Orchestration Module – Medical Chatbot
Coordinates the retrieval-augmented generation (RAG) pipeline:
1. Receive user query (+ optional attachment: image / PDF / data / text)
2. Generate query embedding
3. Retrieve relevant passages from vector store
4. Generate context-aware response via LLM
   - Gemini (multimodal) when GEMINI_API_KEY is set
   - Local flan-t5-small fallback otherwise (text-only)
5. Return response with source information and the answer basis
   (knowledge base / general AI knowledge / both)
"""

import logging
import time
from typing import Dict, List, Optional

from src.embeddings import generate_single_embedding
from src.llm import generate_response, get_model_info
from src import gemini_client

logger = logging.getLogger(__name__)

# Cosine-similarity cutoff for treating a KB passage as relevant to the query.
RELEVANCE_THRESHOLD = 0.40


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
    
    def process_message(self, user_message: str, attachment: Optional[Dict] = None) -> Dict:
        """
        Process a user message through the full RAG pipeline.
        
        Args:
            user_message: The user's input message (may be empty if a file is attached).
            attachment: Optional parsed upload from file_processor.process_upload.
        
        Returns:
            Dictionary with response, sources, basis, engine and metadata.
        """
        start_time = time.time()
        user_message = (user_message or "").strip()
        
        # Validate input
        if not user_message and not attachment:
            return {
                "response": "Please enter a medical question or attach a file.",
                "sources": [],
                "processing_time": 0
            }
        
        attachment_info = None
        if attachment:
            attachment_info = {
                "kind": attachment["kind"],
                "filename": attachment["filename"],
                "preview": attachment["preview"],
            }
        
        try:
            # Step 1-2: Retrieve KB passages. Use the question plus a slice of any
            # extracted attachment text so e.g. a lab report pulls in relevant topics.
            retrieval_query = user_message
            if attachment and attachment.get("text"):
                retrieval_query = f"{user_message}\n{attachment['text'][:1000]}".strip()
            retrieved = []
            if retrieval_query:
                query_embedding = generate_single_embedding(retrieval_query)
                retrieved = self.vector_store.query(
                    query_embedding=query_embedding,
                    top_k=self.top_k
                )
            relevant = [
                r for r in retrieved
                if r.get("text") and r.get("score", 0) >= RELEVANCE_THRESHOLD
            ]
            
            # Step 3-4: Generate the answer.
            engine = "local"
            if gemini_client.is_available():
                try:
                    response_text, basis = gemini_client.generate(
                        query=user_message,
                        kb_passages=relevant,
                        history=self.conversation_history,
                        attachment=attachment,
                    )
                    engine = "gemini"
                except RuntimeError as e:
                    logger.warning(f"Gemini failed, using local fallback: {e}")
                    response_text, basis = self._local_answer(user_message, retrieved, attachment)
                    response_text = f"⚠️ {e} Showing a basic local answer instead.\n\n{response_text}"
            else:
                response_text, basis = self._local_answer(user_message, retrieved, attachment)
            
            # Step 5: Build sources list (only passages the answer could have used).
            source_pool = relevant if engine == "gemini" else retrieved
            sources = []
            if basis != "GENERAL":
                for i, r in enumerate(source_pool[:3]):
                    sources.append({
                        "label": f"KB{i + 1}",
                        "text": r["text"][:150] + ("..." if len(r["text"]) > 150 else ""),
                        "score": round(r.get("score", 0), 4),
                        "id": r.get("id", "")
                    })
            
            processing_time = round(time.time() - start_time, 2)
            
            # Store in conversation history (attachment noted for follow-ups).
            history_user = user_message or "(sent a file)"
            if attachment_info:
                history_user += f"\n[Attached {attachment_info['kind']}: {attachment_info['filename']}]"
            self.conversation_history.append({
                "user": history_user,
                "assistant": response_text,
                "sources": sources,
                "timestamp": time.time(),
                "processing_time": processing_time
            })
            
            return {
                "response": response_text,
                "sources": sources,
                "basis": basis,
                "engine": engine,
                "attachment": attachment_info,
                "processing_time": processing_time
            }
        
        except Exception as e:
            logger.exception(f"Error processing message: {e}")
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
    
    def _local_answer(self, user_message: str, retrieved: List[Dict], attachment: Optional[Dict]):
        """
        Text-only fallback using the local flan-t5-small model.
        
        Returns:
            (response_text, basis) – basis is "KB" (answers come from retrieved passages).
        """
        sections = []
        if attachment:
            kind = attachment["kind"]
            if kind == "image":
                sections.append(
                    "📷 **Image analysis needs the Gemini AI model.** The local model can only read text. "
                    "Add a free `GEMINI_API_KEY` to the `.env` file and restart the app to enable "
                    "image understanding."
                )
            elif kind == "data":
                # The pandas summary is already a useful, factual analysis on its own.
                summary = attachment["text"].split("All rows:")[0].split("First 30 rows:")[0].strip()
                sections.append(f"📊 **Data summary of {attachment['filename']}:**\n\n{summary}")
            elif attachment.get("text"):
                snippet = attachment["text"][:600].strip()
                sections.append(f"📄 **Text read from {attachment['filename']}:**\n\n{snippet}...")
            else:
                sections.append(
                    "📄 This PDF has no text layer (it's probably scanned). Reading scanned documents "
                    "needs the Gemini AI model – add a `GEMINI_API_KEY` to `.env`."
                )
        
        if user_message:
            context_passages = [r["text"] for r in retrieved if r.get("text")]
            sections.append(generate_response(query=user_message, context_passages=context_passages))
        
        return "\n\n".join(sections), "KB"
    
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
        gemini_ready = gemini_client.is_available()
        return {
            "vector_store_count": self.vector_store.get_count(),
            "conversation_length": len(self.conversation_history),
            "engine": "gemini" if gemini_ready else "local",
            "multimodal": gemini_ready,
            "gemini": gemini_client.get_info(),
            "model_info": get_model_info(),
            "top_k": self.top_k
        }
