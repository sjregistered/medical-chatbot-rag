"""
LLM Module – Medical Chatbot
Handles language model integration for generating context-aware medical responses.

Architecture:
- Primary: HuggingFace transformers pipeline (flan-t5-small)
- The PDF specifies Llama-2-7B-Chat-GGML, but that requires ~8GB+ RAM.
  Using a lightweight model as a practical fallback. See README for details.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Global model instances (lazy-loaded)
_pipeline = None  # tuple of (tokenizer, model) once loaded
_model_name = "google/flan-t5-small"


def get_pipeline():
    """
    Lazy-load the seq2seq tokenizer and model.

    Uses tokenizer + model.generate() directly instead of the
    `text2text-generation` pipeline, which was removed in transformers v5.

    Returns:
        Tuple of (tokenizer, model).
    """
    global _pipeline
    if _pipeline is None:
        logger.info(f"Loading LLM: {_model_name}")
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        tokenizer = AutoTokenizer.from_pretrained(_model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(_model_name)
        model.eval()
        _pipeline = (tokenizer, model)
        logger.info("LLM loaded successfully.")
    return _pipeline


def _generate(prompt: str, max_new_tokens: int = 200) -> str:
    """Run generation for a single prompt and return decoded text."""
    import torch

    tokenizer, model = get_pipeline()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def generate_response(
    query: str,
    context_passages: List[str],
    max_context_length: int = 1500
) -> str:
    """
    Generate a medical response using retrieved context passages.
    
    Constructs a prompt with relevant context and sends it to the LLM
    for context-aware answer generation.
    
    Args:
        query: The user's medical question.
        context_passages: List of relevant text passages from the vector store.
        max_context_length: Maximum characters of context to include.
    
    Returns:
        Generated response string.
    """
    # Build context from retrieved passages
    context = ""
    for i, passage in enumerate(context_passages):
        addition = f"\n[Source {i+1}]: {passage}"
        if len(context) + len(addition) > max_context_length:
            break
        context += addition
    
    # Construct the prompt
    prompt = _build_medical_prompt(query, context)
    
    if not context_passages:
        return _fallback_response(query, context_passages)

    try:
        response = _generate(prompt)
        # flan-t5-small sometimes produces very short/degenerate answers;
        # fall back to the retrieved passages in that case.
        if response and len(response.split()) >= 3:
            return response

        return _fallback_response(query, context_passages)
    
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        return _fallback_response(query, context_passages)


def _build_medical_prompt(query: str, context: str) -> str:
    """
    Build a structured prompt for medical question answering.
    
    Args:
        query: User's question.
        context: Retrieved context passages.
    
    Returns:
        Formatted prompt string.
    """
    prompt = (
        f"You are a helpful medical information assistant. "
        f"Answer the following medical question using the provided context. "
        f"Be accurate, concise, and helpful. "
        f"If the context doesn't contain enough information, say so.\n\n"
        f"Context:{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    return prompt


def _fallback_response(query: str, context_passages: List[str]) -> str:
    """
    Generate a fallback response using retrieved context directly.
    Used when the LLM pipeline fails or produces empty output.
    
    Args:
        query: User's question.
        context_passages: Retrieved passages.
    
    Returns:
        A formatted response based on retrieved passages.
    """
    if not context_passages:
        return (
            "I apologize, but I couldn't find relevant medical information "
            "for your query. Please try rephrasing your question or ask about "
            "a specific medical condition, treatment, or symptom."
        )
    
    response = "Based on the available medical knowledge:\n\n"
    for i, passage in enumerate(context_passages[:3]):
        snippet = passage.strip()
        response += f"• {snippet}\n\n"
    
    response += (
        "\n*Note: This information is for educational purposes only. "
        "Please consult a healthcare professional for medical advice.*"
    )
    return response


def get_model_info() -> dict:
    """
    Return information about the current LLM configuration.
    
    Returns:
        Dictionary with model details.
    """
    return {
        "model_name": _model_name,
        "model_type": "text2text-generation",
        "framework": "HuggingFace Transformers",
        "note": (
            "Using flan-t5-small as lightweight fallback. "
            "PDF specifies Llama-2-7B-Chat-GGML which requires ~8GB+ RAM."
        ),
        "loaded": _pipeline is not None
    }
