# Medical Chatbot – AI Health Assistant

> RTB Cycle 16 – Generative AI Advance (UI/UX Specialization)  
> Case Study: Medical Chatbot with RAG Pipeline

## Overview

A production-quality medical chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer medical questions. The system ingests a medical knowledge corpus, creates vector embeddings, stores them in a vector database, and retrieves relevant passages to generate context-aware responses.

With a (free) Gemini API key it becomes **multimodal**: it can read **photos** (skin, X-rays, prescriptions, medicine labels), **PDF lab reports** (including scanned ones), **CSV/Excel health data**, and **text files**, and gives much more intelligent answers.

## Architecture

```
User Query (+ optional file) → File Parsing → Embedding → Vector Search → Relevance Filter
        → Gemini (multimodal) or local flan-t5 fallback → Response + basis badge + sources
```

### Components

| Module | Description |
|--------|-------------|
| `src/ingestion.py` | Loads medical corpus, chunks text with overlap |
| `src/embeddings.py` | Generates sentence embeddings (all-MiniLM-L6-v2) |
| `src/vector_store.py` | Pinecone (primary) + Chroma (fallback) vector DB |
| `src/file_processor.py` | Validates & parses uploads (image / PDF / CSV-Excel / text) |
| `src/gemini_client.py` | Gemini multimodal LLM (images, PDFs, conversation memory) |
| `src/llm.py` | Local flan-t5-small fallback (text-only, no API key needed) |
| `src/chat.py` | RAG pipeline orchestration + engine routing |
| `template.py` | Project structure generator (modular backend) |
| `app.py` | Flask application entry point |

## Setup & Installation

### Prerequisites
- Python 3.11+
- pip

### Steps

```bash
# 1. Navigate to project
cd medical-chatbot

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Enable multimodal AI (recommended)
cp .env.example .env
# Edit .env and set GEMINI_API_KEY (free key: https://aistudio.google.com/apikey)
# Optionally add PINECONE_API_KEY

# 5. Run the application
python app.py
```

The app will be available at **http://localhost:5050**. The header badge shows the active engine
(**✨ Gemini** = multimodal, **⚙️ Local model** = text-only fallback).

## Uploading Files

Pick what you're uploading from the dropdown next to the 📎 button (or leave **Auto-detect**), then attach via
the button, drag & drop, or paste. Max 10 MB.

| Dropdown option | File types | What the bot does |
|---|---|---|
| 📷 Photo / X-ray | PNG, JPG, WEBP, GIF, HEIC | Describes visible findings, possible causes, reads labels/prescriptions *(Gemini only)* |
| 📄 PDF report | PDF | Extracts values, compares with reference ranges, flags high/low |
| 📊 Data (CSV/Excel) | CSV, XLSX, XLS | Computes stats & trends (pandas), then interprets them |
| 📝 Text file | TXT, MD | Reads and explains the document |

Each answer shows a **basis badge**: 📚 *Knowledge base*, 🧠 *AI general knowledge*, or 🔗 *Knowledge base + AI*,
with inline `[KB1]` citations when the knowledge base was used.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve the chat UI |
| POST | `/chat` | Send a message and get a response |
| GET | `/history` | Retrieve conversation history |
| POST | `/clear` | Clear conversation history |
| GET | `/status` | Check system status |

### Chat API Example

```bash
curl -X POST http://localhost:5050/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the symptoms of diabetes?"}'

# With a file (multipart): kind = auto | image | pdf | data | text
curl -X POST http://localhost:5050/chat \
  -F "message=Explain my results" -F "kind=pdf" -F "file=@lab_report.pdf"
```

## Deviations from PDF Specification

| Specified | Implemented | Reason |
|-----------|------------|--------|
| Llama-2-7B-Chat-GGML | Gemini 2.5 Flash (multimodal) + google/flan-t5-small fallback | Llama-2-7B requires ~8GB+ RAM and can't read images; Gemini adds vision/PDF understanding, flan-t5 keeps the app working offline without a key |
| Pinecone (only) | Pinecone + Chroma fallback | Graceful degradation when no API key is available; Chroma provides identical functionality locally |
| Gale Encyclopedia PDF | Custom medical corpus | Representative medical knowledge covering 12 conditions with equivalent depth |

## Features

- **RAG Pipeline**: Full retrieval-augmented generation with vector similarity search
- **Multimodal Input**: Photos, PDF reports, CSV/Excel data and text files (dropdown-selected type)
- **Hybrid Answers**: Uses the knowledge base when relevant, the AI's own knowledge otherwise, and labels which
- **Conversation Memory**: Follow-up questions understand earlier messages and attachments
- **Premium Chat UI**: Dark mode, message bubbles, typing indicators, drag & drop, suggestion chips
- **Conversation History**: Sidebar with chat history tracking
- **Source Citations**: Expandable source references with relevance scores
- **Responsive Design**: Works on desktop and mobile
- **Modular Architecture**: Clean separation of concerns per `template.py`

## Tech Stack

- **Backend**: Python 3.11+, Flask
- **LLM**: Google Gemini (multimodal) / HuggingFace Transformers flan-t5-small (fallback)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: Pinecone / ChromaDB
- **File parsing**: pypdf, pandas, openpyxl, Pillow
- **Frontend**: HTML, CSS, Vanilla JavaScript
