"""
Gemini Client Module – Medical Chatbot
Multimodal LLM backend (text + images + PDFs) using the Google Gemini API.

Enabled when GEMINI_API_KEY (or GOOGLE_API_KEY) is set in the environment / .env.
If no key is configured, `is_available()` returns False and the chatbot falls
back to the local flan-t5-small model in `src/llm.py` (text-only).
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.8-flash"

_client = None
_init_error: Optional[str] = None

SYSTEM_INSTRUCTION = """You are MedBot, a careful, knowledgeable medical information assistant.

How to answer:
- Give clear, well-structured, accurate answers. Use short paragraphs, **bold** key terms, and "- " bullet lists where helpful.
- When KNOWLEDGE BASE passages are provided and relevant, ground your answer in them and cite them inline as [KB1], [KB2], etc.
- If the passages are missing or don't cover the question, answer from your own general medical knowledge and do not invent citations.
- When the user attaches a file:
  * Images (skin, X-ray, scan, wound, prescription, medicine label): describe what you can actually see, list possible explanations from most to least likely, state what can't be judged from a photo, and suggest next steps. Read any visible text (drug names, doses).
  * Lab reports / PDFs: extract the key values, compare them with typical reference ranges (use the report's own ranges if printed), flag values that are high/low, and explain in plain language what they may indicate.
  * Data files (CSV/Excel): analyze trends, outliers and out-of-range values, and summarize the important findings with numbers.
- Never give a definitive diagnosis. Explain possibilities and say when a clinician should be seen.
- If anything suggests an emergency (chest pain, stroke signs, severe bleeding, difficulty breathing, suicidal thoughts, very abnormal critical lab values), say so first and advise urgent care.
- Keep answers focused: usually 120-350 words unless the user asks for more detail.

At the very end of your reply, on its own line, output exactly one tag describing what your answer was based on:
[[BASIS: KB]]       – mainly the knowledge base passages
[[BASIS: GENERAL]]  – mainly your own general medical knowledge
[[BASIS: BOTH]]     – a meaningful mix of both
"""

_BASIS_RE = re.compile(r"\[\[\s*BASIS\s*:\s*(KB|GENERAL|BOTH)\s*\]\]\s*$", re.IGNORECASE)


def _api_key() -> Optional[str]:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)


def is_available() -> bool:
    """True if a Gemini API key is configured and the SDK could be initialized."""
    return _get_client() is not None


def _get_client():
    global _client, _init_error
    if _client is not None:
        return _client
    key = _api_key()
    if not key:
        return None
    try:
        from google import genai
        _client = genai.Client(api_key=key)
        _init_error = None
        logger.info(f"Gemini client initialized (model={model_name()})")
    except Exception as e:  # SDK missing or bad config
        _init_error = str(e)
        logger.error(f"Failed to initialize Gemini client: {e}")
        _client = None
    return _client


def generate(
    query: str,
    kb_passages: List[Dict],
    history: List[Dict],
    attachment: Optional[Dict] = None,
) -> Tuple[str, str]:
    """
    Generate an answer with Gemini.

    Args:
        query: The user's message (may be empty if only a file was sent).
        kb_passages: Relevant knowledge-base passages: [{"text", "score"}, ...].
        history: Prior turns: [{"user": str, "assistant": str}, ...].
        attachment: Optional attachment dict from file_processor.process_upload.

    Returns:
        (answer_text, basis) where basis is "KB", "GENERAL" or "BOTH".

    Raises:
        RuntimeError: With a user-safe message if the API call fails.
    """
    from google.genai import types, errors

    client = _get_client()
    if client is None:
        raise RuntimeError("Gemini is not configured.")

    contents = []
    # Recent conversation for follow-up questions ("what about the second value?").
    for turn in history[-6:]:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["user"])]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=turn["assistant"])]))

    parts = []
    if attachment:
        kind = attachment["kind"]
        if kind in ("image", "pdf"):
            # Send the raw file: Gemini reads images and PDFs (incl. scanned ones) natively.
            parts.append(types.Part.from_bytes(data=attachment["bytes"], mime_type=attachment["mime"]))
        if attachment.get("text"):
            label = {
                "pdf": "Extracted text from the attached PDF",
                "data": "Parsed contents and statistics of the attached data file",
                "text": "Contents of the attached text file",
            }.get(kind, "Attachment text")
            parts.append(types.Part.from_text(
                text=f"{label} ('{attachment['filename']}'):\n{attachment['text']}"
            ))

    if kb_passages:
        kb_block = "\n\n".join(
            f"[KB{i + 1}] (relevance {p['score']:.2f}): {p['text']}" for i, p in enumerate(kb_passages)
        )
    else:
        kb_block = "(no sufficiently relevant passages found)"

    if not query:
        query = {
            "image": "Please analyze this image and explain what it may show.",
            "pdf": "Please review this document and explain the key findings.",
            "data": "Please analyze this data and summarize the important findings.",
            "text": "Please review this document and explain the key points.",
        }.get(attachment["kind"] if attachment else "", "Please help.")

    parts.append(types.Part.from_text(
        text=f"KNOWLEDGE BASE PASSAGES:\n{kb_block}\n\nUSER QUESTION:\n{query}"
    ))
    contents.append(types.Content(role="user", parts=parts))

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
        max_output_tokens=2048,
    )

    try:
        response = client.models.generate_content(model=model_name(), contents=contents, config=config)
    except errors.ClientError as e:
        logger.error(f"Gemini client error: {e}")
        code = getattr(e, "code", None)
        if code in (401, 403) or "API key" in str(e):
            raise RuntimeError("The Gemini API key is invalid or lacks permission. Check GEMINI_API_KEY in .env.")
        if code == 429:
            raise RuntimeError("Gemini rate limit reached. Please wait a minute and try again.")
        if code == 404:
            raise RuntimeError(f"Gemini model '{model_name()}' was not found. Set GEMINI_MODEL in .env to an available model.")
        raise RuntimeError("Gemini rejected the request (the file may be unsupported or too large).")
    except errors.APIError as e:
        logger.error(f"Gemini API error: {e}")
        raise RuntimeError("The Gemini service is temporarily unavailable. Please try again.")
    except Exception as e:
        logger.error(f"Gemini call failed: {e}")
        raise RuntimeError("Could not reach the Gemini service. Check your internet connection.")

    text = (getattr(response, "text", None) or "").strip()
    if not text:
        raise RuntimeError("Gemini returned an empty response (it may have been blocked by safety filters).")

    basis = "BOTH" if kb_passages else "GENERAL"
    match = _BASIS_RE.search(text)
    if match:
        basis = match.group(1).upper()
        text = text[: match.start()].rstrip()
    if basis in ("KB", "BOTH") and not kb_passages:
        basis = "GENERAL"  # can't be KB-based without passages
    return text, basis


def get_info() -> Dict:
    return {
        "provider": "Google Gemini",
        "model_name": model_name(),
        "configured": bool(_api_key()),
        "ready": _client is not None,
        "error": _init_error,
        "multimodal": True,
    }
