# RAG Chatbot System

A production-grade Retrieval-Augmented Generation (RAG) chatbot that answers questions based on your uploaded documents using hybrid search, cross-encoder re-ranking, and LLM generation.

## Features

- **Multi-format ingestion** — PDF, Markdown, HTML, TXT
- **Advanced chunking** — Semantic (sentence-level via spaCy/NLTK) + parent-child chunking for better context preservation
- **Hybrid search** — Dense vector search (cosine similarity) combined with keyword/BM25 matching
- **Cross-encoder re-ranking** — BGE-reranker-v2-m3 for precision
- **Query rewriting** — LLM-based standalone query generation using chat history
- **Chat memory** — Session-based conversation history
- **Source citations** — Every answer references specific documents
- **Vietnamese support** — Bilingual UI and instructions

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| LLM | DeepSeek (OpenAI-compatible API) |
| Embedding | sentence-transformers (BAAI/bge-m3) or OpenAI API |
| Vector DB | Qdrant |
| Re-ranker | BAAI/bge-reranker-v2-m3 (Cross-Encoder) |
| Parsing | PyMuPDF, BeautifulSoup4 |
| Eval | RAGAS, pytest |

## Quick Start

### Prerequisites

- Docker and Docker Compose
- A DeepSeek API key (set in `backend/.env`)

### Setup

```bash
# 1. Clone and navigate
cd chatbox-rag

# 2. Set your API key
echo 'LLM_API_KEY=sk-your-key-here' > backend/.env

# 3. Start the services
docker compose up -d --build
```

The app will be available at **http://localhost:8000**.

### Manual Setup (without Docker)

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check (Qdrant, embedder, reranker status) |
| `GET` | `/api/v1/info` | API info and endpoint listing |
| `POST` | `/api/v1/ingest` | Upload and process a document file |
| `POST` | `/api/v1/ingest/text` | Ingest raw text |
| `GET` | `/api/v1/documents` | List all ingested documents |
| `DELETE` | `/api/v1/documents/{id}` | Delete a document and its chunks |
| `POST` | `/api/v1/chat` | Send a query and get an AI response |
| `GET` | `/api/v1/history/{session_id}` | Get chat history for a session |
| `DELETE` | `/api/v1/history/{session_id}` | Clear chat history for a session |

Swagger docs: **http://localhost:8000/api/v1/docs**

## Environment Variables

See `backend/.env.example` for all options. Key variables:

| Variable | Default | Description |
|---|---|---|
| `LLM_API_KEY` | — | DeepSeek API key (required) |
| `LLM_BASE_URL` | `https://api.deepseek.com` | LLM API endpoint |
| `LLM_MODEL` | `deepseek-chat` | Model name |
| `EMBEDDING_PROVIDER` | `local` | `local` (sentence-transformers) or `api` |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection URL |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Re-ranker model |

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────┐
│  Frontend   │────▶│  FastAPI      │────▶│ Qdrant  │
│  (HTML/JS)  │     │  Backend      │     │   DB    │
└─────────────┘     │               │     └─────────┘
                    │  ┌─────────┐  │
                    │  │Embedder │  │     ┌─────────┐
                    │  │Reranker │──┼────▶│ HFace   │
                    │  │ Chunker │  │     │ Models  │
                    │  └─────────┘  │     └─────────┘
                    │               │
                    │  ┌─────────┐  │     ┌─────────┐
                    │  │  LLM    │──┼────▶│DeepSeek │
                    │  └─────────┘  │     │   API   │
                    └──────────────┘     └─────────┘
```

### Retrieval Pipeline

1. **Query Rewriting** — LLM reformulates the user query into a standalone question using chat history
2. **Hybrid Search** — Combines vector search (70%) + keyword/BM25 (30%) from Qdrant
3. **Re-ranking** — Cross-encoder scores and re-ranks the top 20 results down to top K
4. **LLM Generation** — Final answer generated with source citations

### Ingestion Pipeline

1. **Load** — PDF/Markdown/HTML/TXT parsing
2. **Preprocess** — Unicode normalization, whitespace cleanup
3. **Chunk** — Semantic sentence splitting + parent-child context expansion
4. **Embed** — Dense vector embedding
5. **Store** — Upsert into Qdrant with metadata indexing

## Project Structure

```
chatbox-rag/
├── docker-compose.yml
├── backend/
│   ├── Dockerfile
│   ├── .env
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py              # FastAPI app
│   │   ├── config.py            # Pydantic settings
│   │   ├── models/              # Data models
│   │   ├── routers/             # API routes
│   │   ├── services/            # Core services
│   │   │   ├── vector_store.py  # Qdrant client
│   │   │   ├── embedder.py      # Embedding service
│   │   │   ├── chunker.py       # Document chunking
│   │   │   ├── retriever.py     # Hybrid search
│   │   │   ├── reranker.py      # Cross-encoder
│   │   │   ├── llm_service.py   # LLM client
│   │   │   ├── query_rewriter.py
│   │   │   ├── chat_memory.py
│   │   │   └── document_loader.py
│   │   └── utils/
│   └── tests/
│       ├── test_api.py
│       └── test_ragas.py
└── frontend/
    ├── index.html
    ├── script.js
    └── style.css
```

## Testing

```bash
cd backend
pip install pytest pytest-asyncio httpx
pytest tests/ -v
```

RAGAS evaluation tests require a running Qdrant instance and valid API keys.

## License

MIT
