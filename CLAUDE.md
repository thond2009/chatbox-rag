# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start all services (Qdrant + backend) with Docker
docker compose up -d --build

# Run backend locally (requires venv, Qdrant on localhost:6333)
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader punkt
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Run tests
cd backend
pytest tests/ -v

# Run RAGAS evaluation (requires OPENAI_API_KEY)
cd backend
python -m tests.test_ragas
```

App at `http://localhost:8000`, Swagger at `http://localhost:8000/api/v1/docs`.

## Architecture

**Retrieval pipeline** (in `routers/chat.py`): Query Rewriting (LLM, uses chat history) → Hybrid Search (vector + BM25) → Cross-Encoder Re-ranking → LLM Generation with system prompt that enforces no-hallucination and source citation.

**Ingestion pipeline** (in `routers/ingestion.py`): file upload → document loader → text preprocessing → semantic sentence chunking with parent-child expansion → embedding → Qdrant upsert.

**Singletons**: `VectorStore`, `ChatMemory`, and `RerankerService` use the singleton pattern (override `__new__`). They are module-level instances imported directly (e.g., `from app.services.vector_store import vector_store`).

**Lazy init**: Embedder and Reranker load models lazily on first use with double-checked locking (`threading.Lock`). Startups pass health checks even before models are warm.

**Hybrid search** (`retriever.py`): merges vector results (Cosine similarity) weighted 0.7 and keyword/BM25 weighted 0.3. Keyword search uses a simple token-match scorer over scrolled Qdrant payloads — not a dedicated sparse index.

**Config**: `config.py` uses pydantic-settings with `.env` override. All tunables (chunk size, weights, top-k) are env-driven with defaults. `EMBEDDING_PROVIDER` switches between `local` (sentence-transformers, BAAI/bge-m3) and `api` (OpenAI-compatible).

**Frontend** is vanilla HTML/JS/CSS served as static files from the FastAPI app (mounted at `/`). No build step.

**Chat memory** is in-process (`defaultdict` keyed by session_id), capped per `MAX_CHAT_HISTORY` × 2 messages. No persistence across restarts.

**System prompts are in Vietnamese** (`llm_service.py`, `query_rewriter.py`). The LLM service uses parent_content (larger context window around each chunk) when available.

## Things to know

- The LLM service is OpenAI-compatible but defaults to DeepSeek API (`https://api.deepseek.com`). Set `LLM_API_KEY` in `backend/.env`.
- Keyword search scrolls up to 1000 points and does token matching in Python — it will degrade with large collections.
- Reranker gracefully degrades: if the model fails to load, it passes through the top-K results unranked.
- Query rewriter gracefully degrades: if the LLM call fails, the original query is used as-is.
