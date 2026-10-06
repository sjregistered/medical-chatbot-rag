"""
app.py – Medical Chatbot Flask Application
RTB Cycle 16 – Generative AI Advance (UI/UX Specialization)

Main Flask application entry point. Initializes the RAG pipeline
(ingestion → embedding → vector store → LLM) and serves the chat API.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from src.ingestion import load_medical_corpus, chunk_text, get_ingestion_stats
from src.embeddings import generate_embeddings, get_embedding_dimension
from src.vector_store import create_vector_store
from src.chat import ChatOrchestrator
from src.file_processor import process_upload, UploadError, MAX_UPLOAD_BYTES

# ─── Configuration ───

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence model-download chatter

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
# Reject oversized uploads before reading them (+1 MB slack for form fields).
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 1024 * 1024

# Global orchestrator instance
orchestrator = None


def initialize_pipeline():
    """
    Initialize the full RAG pipeline:
    1. Load medical corpus
    2. Chunk text
    3. Generate embeddings
    4. Store in vector database
    5. Create chat orchestrator
    """
    global orchestrator
    
    logger.info("=" * 60)
    logger.info("Initializing Medical Chatbot RAG Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Create vector store
    logger.info("Step 1: Initializing vector store...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    vector_store = create_vector_store(
        collection_name="medical_chatbot",
        persist_directory=os.path.join(base_dir, "chroma_db")
    )
    
    # Step 2: Check if data already ingested
    existing_count = vector_store.get_count()
    if existing_count > 0:
        logger.info(
            f"Vector store already contains {existing_count} documents. "
            "Skipping ingestion."
        )
    else:
        # Step 3: Load and chunk corpus
        logger.info("Step 2: Loading medical corpus...")
        data_dir = os.path.join(base_dir, "data")
        corpus = load_medical_corpus(data_dir)
        logger.info(f"Loaded corpus: {len(corpus)} characters")
        
        logger.info("Step 3: Chunking text...")
        chunks = chunk_text(corpus, chunk_size=500, chunk_overlap=50)
        stats = get_ingestion_stats(chunks)
        logger.info(f"Chunking complete: {stats}")
        
        # Step 4: Generate embeddings
        logger.info("Step 4: Generating embeddings...")
        texts = [c["text"] for c in chunks]
        embeddings = generate_embeddings(texts)
        dim = get_embedding_dimension()
        logger.info(f"Generated {len(embeddings)} embeddings (dim={dim})")
        
        # Step 5: Store in vector database
        logger.info("Step 5: Storing in vector database...")
        ids = [c["id"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        vector_store.add_documents(ids, texts, embeddings, metadatas)
        logger.info(f"Stored {len(ids)} documents in vector store.")
    
    # Step 6: Create orchestrator
    logger.info("Step 6: Creating chat orchestrator...")
    orchestrator = ChatOrchestrator(vector_store=vector_store, top_k=5)

    from src import gemini_client
    if gemini_client.is_available():
        logger.info(f"Gemini enabled (model={gemini_client.model_name()}) – multimodal answers ON")
        logger.info("Skipping local LLM (flan-t5) memory load since Gemini will be used.")
    else:
        logger.info("No GEMINI_API_KEY set – using local text-only model (images need Gemini)")
        # Step 7: Warm up the LLM so the first request is fast (non-fatal)
        try:
            logger.info("Step 7: Loading local fallback LLM (flan-t5)...")
            from src.llm import get_pipeline
            get_pipeline()
        except Exception as e:
            logger.warning(f"LLM failed to load ({e}); responses will use retrieval fallback.")
    
    logger.info("=" * 60)
    logger.info("Medical Chatbot RAG Pipeline READY")
    logger.info("=" * 60)


# ─── Routes ───

@app.route("/")
def index():
    """Serve the main chat interface."""
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chat messages, optionally with an attached file.
    
    Accepts either:
      - JSON: { "message": "user question" }
      - multipart/form-data: message=<text>, file=<upload>, kind=auto|image|pdf|data|text
    Returns JSON: { "response", "sources", "basis", "engine", "attachment", "processing_time" }
    """
    if orchestrator is None:
        return jsonify({
            "response": "The chatbot is still initializing. Please wait a moment.",
            "sources": [],
            "processing_time": 0
        }), 503
    
    attachment = None
    if request.content_type and request.content_type.startswith("multipart/form-data"):
        user_message = (request.form.get("message") or "").strip()[:2000]
        upload = request.files.get("file")
        if upload and upload.filename:
            try:
                attachment = process_upload(
                    filename=upload.filename,
                    data=upload.read(),
                    kind=request.form.get("kind", "auto"),
                )
            except UploadError as e:
                return jsonify({"response": str(e), "sources": [], "processing_time": 0}), 400
    else:
        data = request.get_json(silent=True)
        if not data or not isinstance(data.get("message"), str):
            return jsonify({
                "response": "Please provide a message.",
                "sources": [],
                "processing_time": 0
            }), 400
        user_message = data["message"].strip()[:2000]
    
    if not user_message and not attachment:
        return jsonify({
            "response": "Please enter a medical question or attach a file.",
            "sources": [],
            "processing_time": 0
        }), 400
    
    result = orchestrator.process_message(user_message, attachment=attachment)
    return jsonify(result)


@app.errorhandler(413)
def too_large(_e):
    """Return a JSON error (instead of an HTML page) for oversized uploads."""
    return jsonify({
        "response": "File is too large. Maximum size is 10 MB.",
        "sources": [],
        "processing_time": 0
    }), 413


@app.route("/history", methods=["GET"])
def history():
    """Return conversation history."""
    if orchestrator is None:
        return jsonify({"history": []}), 503
    
    return jsonify({"history": orchestrator.get_history()})


@app.route("/clear", methods=["POST"])
def clear():
    """Clear conversation history."""
    if orchestrator is None:
        return jsonify({"status": "error"}), 503
    
    orchestrator.clear_history()
    return jsonify({"status": "cleared"})


@app.route("/status", methods=["GET"])
def status():
    """Return system status information."""
    if orchestrator is None:
        return jsonify({"status": "initializing"}), 503
    
    return jsonify(orchestrator.get_status())


# ─── Main ───

# Initialize pipeline globally so it runs for WSGI servers like Gunicorn
try:
    initialize_pipeline()
except Exception as e:
    # Keep the server up so the UI loads and /status reports the problem
    logger.exception(f"Pipeline initialization failed: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Medical Chatbot on port {port}")
    # use_reloader=False prevents loading the models twice in debug mode
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
