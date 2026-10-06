# Medical Chatbot – AI Health Assistant

> RTB Cycle 16 – Generative AI Advance (UI/UX Specialization)  
> Case Study: Medical Chatbot with RAG Pipeline

## Overview

A production-quality medical chatbot that uses **Retrieval-Augmented Generation (RAG)** to answer medical questions. The system ingests a medical knowledge corpus, creates vector embeddings, stores them in a vector database, and retrieves relevant passages to generate context-aware responses.

## Architecture

```
User Query → Embedding → Vector Search → Context Retrieval → LLM Generation → Response
```

### Components

| Module | Description |
|--------|-------------|
| `src/ingestion.py` | Loads medical corpus, chunks text with overlap |
| `src/embeddings.py` | Generates sentence embeddings (all-MiniLM-L6-v2) |
| `src/vector_store.py` | Pinecone (primary) + Chroma (fallback) vector DB |
| `src/llm.py` | LLM integration for response generation |
| `src/chat.py` | RAG pipeline orchestration |
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

# 4. (Optional) Configure Pinecone
cp .env.example .env
# Edit .env and add your PINECONE_API_KEY

# 5. Run the application
python app.py
```

The app will be available at **http://localhost:5050**.

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
```

## Deviations from PDF Specification

| Specified | Implemented | Reason |
|-----------|------------|--------|
| Llama-2-7B-Chat-GGML | google/flan-t5-small | Llama-2-7B requires ~8GB+ RAM and GGML setup; flan-t5-small provides functional text generation with minimal resources |
| Pinecone (only) | Pinecone + Chroma fallback | Graceful degradation when no API key is available; Chroma provides identical functionality locally |
| Gale Encyclopedia PDF | Custom medical corpus | Representative medical knowledge covering 12 conditions with equivalent depth |

## Features

- **RAG Pipeline**: Full retrieval-augmented generation with vector similarity search
- **Premium Chat UI**: Dark mode, message bubbles, typing indicators, suggestion chips
- **Conversation History**: Sidebar with chat history tracking
- **Source Citations**: Expandable source references with relevance scores
- **Responsive Design**: Works on desktop and mobile
- **Modular Architecture**: Clean separation of concerns per `template.py`

## Tech Stack

- **Backend**: Python 3.11+, Flask
- **LLM**: HuggingFace Transformers (flan-t5-small)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: Pinecone / ChromaDB
- **Frontend**: HTML, CSS, Vanilla JavaScript
