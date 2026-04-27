# Learning AI Through This RAG Project

This guide walks through the AI concepts implemented in this codebase, from first principles to production patterns. Read it alongside the source code.

---

## 1. What Problem Does RAG Solve?

LLMs know only what they were trained on. They hallucinate (invent facts) and can't access your private documents.

**RAG** fixes this by giving the LLM relevant excerpts from your documents at query time. Instead of asking the LLM "from memory", you ask it "based on these documents."

The full pipeline in this project:

```
Your question → Rewrite for clarity → Search documents → Re-rank results → Feed top results to LLM → Answer with citations
```

Each step is a distinct AI concept explained below.

---

## 2. Embeddings: Turning Text Into Numbers

**File:** `backend/app/services/embedder.py`

### Concept

Computers can't compare text meaningfully. "Dog" and "puppy" look completely different as strings. Embeddings solve this by converting text into a list of numbers (a vector) where similar meanings produce similar vectors.

```
"Con chó đang chạy" → [0.23, -0.45, 0.78, ..., 0.12]  (1024 numbers)
"Mèo đang ngủ"      → [-0.31, 0.52, -0.19, ..., 0.67]  (very different vector)
"Chó con đang chạy" → [0.21, -0.41, 0.75, ..., 0.15]  (close to first vector)
```

### How this project does it

Two modes (`EMBEDDING_PROVIDER` env var):

| Mode | What happens | Trade-off |
|------|-------------|-----------|
| `local` | Downloads BAAI/bge-m3 model, runs on your CPU/GPU | No API cost, needs RAM, supports Vietnamese |
| `api` | Calls OpenAI-compatible embedding API | No local compute, costs per call |

**Key detail in `embedder.py:60-63`:** The `_ensure_model()` method uses double-checked locking (`threading.Lock`). The model loads lazily — only on first use, not at import time. This is critical in production because FastAPI starts before models are warm (health checks still pass).

**Key detail in `embedder.py:71`:** Local embeddings truncate text to 8000 characters. Most embedding models have a token limit; exceeding it silently breaks quality.

### What to learn

- Embeddings are the foundation of all modern search. Without them, you're doing Ctrl+F.
- Model choice matters: BGE-M3 is multilingual (Vietnamese + English), which is why this project picked it.
- Normalization (`normalize_embeddings=True`) makes cosine similarity work correctly.

---

## 3. Vector Search: Finding Relevant Documents

**Files:** `backend/app/services/vector_store.py`, `backend/app/services/retriever.py`

### Concept

Once every document chunk is converted to a vector, searching becomes a math problem: find the K vectors closest to your query vector. Qdrant stores these vectors and does the nearest-neighbor search efficiently (using HNSW index under the hood).

Cosine similarity measures the angle between two vectors. Parallel vectors (small angle) = similar meaning. Perpendicular vectors = unrelated.

### How this project does it

**File `vector_store.py:111-138`:** The `search()` method sends a query embedding to Qdrant and gets back scored payloads. The score is cosine distance (closer to 1 = more similar).

**But vector search alone misses things.** If a user searches for "error code E501", a vector model might match it to "troubleshooting connection issues" (semantically similar) but miss the exact string "E501" in a log file.

### Hybrid Search

**File `retriever.py:10-54`:** This project combines two search methods:

1. **Vector search** (semantic) — understands meaning: "con chó" ≈ "animal companion"
2. **Keyword search/BM25** (lexical) — exact string matching: finds "E501" precisely

The merge formula (`VECTOR_WEIGHT=0.7`, `KEYWORD_WEIGHT=0.3`):
```
final_score = vector_score × 0.7 + keyword_score × 0.3
```

**Key detail in `retriever.py:23`:** It fetches `top_k * 2` results from each method, then merges and returns only `top_k`. Fetching extra results before merging prevents good keyword matches from being crowded out.

**Key detail in `vector_store.py:140-166`:** The keyword search is a simple token-match scorer in Python, not a proper BM25. It scrolls all points from Qdrant (capped at 1000) and counts token overlap. This is a **known scalability limit** — with millions of chunks, you'd need Qdrant's sparse vector index instead.

### What to learn

- Vector search understands intent; keyword search catches exact terms. Neither is sufficient alone.
- The weights (0.7/0.3) are a hyperparameter — tune them based on your data. Code-heavy docs need higher keyword weight.
- This is called **HyDE**-like or **fusion retrieval**. The general pattern is: fetch from multiple sources, merge scores, return top.

---

## 4. Re-ranking: Making Search Results Precise

**File:** `backend/app/services/reranker.py`

### Concept

The initial search (vector + keyword) is fast but coarse. It looks at each document in isolation. A **cross-encoder** re-ranker reads the query and each document *together* and produces a relevance score. This is slower but much more accurate.

Think of it as:
- **Bi-encoder (initial search):** encodes query and documents separately, compares quickly. Like scanning book titles on a shelf.
- **Cross-encoder (re-ranker):** reads query + document as a pair, judges relevance. Like actually reading the paragraph to see if it answers the question.

### How this project does it

**The pipeline in `chat.py:27-31`:**
```python
search_results = hybrid_search(query=rewritten_query, top_k=20)  # Get 20 candidates
reranked_results = reranker_service.rerank(query, documents=search_results, top_k=5)  # Pick best 5
```

Fetch 20, keep 5. This is the standard "funnel" pattern in RAG.

**File `reranker.py:49-86`:** The cross-encoder takes `[query, document_content]` pairs, runs them through a BERT-like model (BAAI/bge-reranker-v2-m3), and outputs a single relevance score per pair.

**Key detail in `reranker.py:66`:** Document content is truncated to 512 characters before re-ranking. Cross-encoders have strict input length limits (typically 512 tokens).

**Key detail in `reranker.py:59-61`:** If the model fails to load (missing dependency, OOM), the reranker **gracefully degrades** — it passes through the top-K results unranked rather than crashing. This is a critical production pattern.

### What to learn

- Re-ranking is the "quality filter" — it costs more compute but dramatically improves precision.
- The number 20→5 is configurable. More initial results = better recall; tighter re-rank = better precision.
- Always design ML components to degrade gracefully. A working system without re-ranking is better than a crashed system.

---

## 5. Chunking: How to Split Documents

**File:** `backend/app/services/chunker.py`

### Concept

Documents are too large to embed directly. Embedding models have token limits, and a 50-page PDF's embedding would be too vague to match anything specific. You must split documents into chunks.

Naive approach: split every N characters. This cuts sentences in half.

Better approach (this project): **semantic sentence splitting** + **parent-child chunking**.

### How this project does it

**Step 1 — Sentence splitting** (`chunker.py:11-29`):
Tries spaCy sentencizer → falls back to NLTK punkt → falls back to regex split. Sentences are never broken mid-way.

**Step 2 — Build chunks from sentences** (`chunker.py:32-65`):
Accumulates sentences until the total length exceeds `CHUNK_SIZE` (default 512 chars), then starts a new chunk. An overlap window (`CHUNK_OVERLAP`, default 50 chars) ensures no context is lost at chunk boundaries.

```
Sentence 1, Sentence 2, Sentence 3 → Chunk A
         Sentence 3, Sentence 4, Sentence 5 → Chunk B  (Sentence 3 overlaps)
```

**Step 3 — Parent-child expansion** (`chunker.py:68-87`):
Each child chunk (512 chars for embedding) gets a parent context (1024 chars surrounding it). The child is used for search (more precise), the parent is fed to the LLM (more context). This is used in `llm_service.py:43`:
```python
content = doc.get("parent_content", doc.get("content", ""))
```

### What to learn

- Chunk size is a critical hyperparameter. Too small = missing context. Too large = embedding too vague.
- Overlap prevents information loss at boundaries.
- Parent-child chunking is a clever pattern: optimize for search AND generation simultaneously, since they have different needs.

---

## 6. Query Rewriting: Understanding Follow-up Questions

**File:** `backend/app/services/query_rewriter.py`

### Concept

Users ask follow-up questions that only make sense with context:

```
User: "Kafka là gì?"
Assistant: "Apache Kafka là một nền tảng streaming..."
User: "Nó chạy trên port bao nhiêu?"  ← "Nó" = Kafka, but the search engine doesn't know that
```

If you search for "Nó chạy trên port bao nhiêu?" you'll get random results. Query rewriting uses an LLM to convert the follow-up into a standalone question: "Apache Kafka chạy trên port bao nhiêu?"

### How this project does it

**File `query_rewriter.py:18-51`:**
- Takes the current query + last 6 chat history messages
- Sends them to the LLM with a rewriting prompt
- Returns the standalone query (or the original if the LLM call fails)

**Key detail in `query_rewriter.py:49-51`:** If rewriting fails, it returns the original query. Graceful degradation again — a slightly worse search is better than no search.

**Key detail in `query_rewriter.py:43`:** Temperature is set to 0.0 for rewriting. You don't want creativity — you want deterministic, consistent reformulation.

### What to learn

- Query rewriting is often the highest-ROI improvement in a RAG system. Multi-turn conversations are unusable without it.
- Use low temperature (0 or 0.1) for any task where consistency matters more than creativity.

---

## 7. Prompt Engineering: Controlling the LLM

**File:** `backend/app/services/llm_service.py`

### Concept

The prompt is how you program an LLM. It defines constraints, format, and behavior. A good RAG prompt prevents hallucination by binding the model to provided context.

### How this project does it

**`llm_service.py:8-24`** — The system prompt (in Vietnamese):

```
You are a technical expert. Answer based SOLELY ON THE DOCUMENTS provided.
CONSTRAINTS:
1. IF information is NOT in the documents, respond exactly: "The current data does not contain this information." Do not speculate.
2. Cite sources (file name) after each point if available.
3. Format using Markdown, use bullet points for readability.
---
RETRIEVED DOCUMENTS: {context}
CHAT HISTORY: {chat_history}
CURRENT QUESTION: {question}
```

This is a **constrained generation** prompt. It explicitly:
- Binds the LLM to provided documents (anti-hallucination)
- Gives a specific "I don't know" response format
- Requires citations (transparency)
- Specifies output format (Markdown with bullets)

**Key detail in `llm_service.py:41-45`:** It uses `parent_content` (larger context) as the document source for the LLM, not the child chunk used for search. The LLM benefits from more context around the matched chunk.

**Key detail in `llm_service.py:64`:** Temperature is 0.1 (configurable). Low temperature = factual, consistent answers. Higher temperature = creative but potentially hallucinated.

### What to learn

- Prompt engineering is the cheapest way to improve RAG quality. A well-written constraint block prevents more hallucinations than any model upgrade.
- The "I don't know" escape hatch is critical. Without it, the LLM will confidently invent facts.
- System prompts should be in the user's language — this project uses Vietnamese because users ask questions in Vietnamese.

---

## 8. LLM API Integration

**Files:** `backend/app/services/llm_service.py`, `backend/app/services/query_rewriter.py`, `backend/app/config.py`

### Concept

You don't need to run LLMs locally. This project calls external LLM APIs using the OpenAI-compatible protocol. This works with OpenAI, DeepSeek, or any compatible provider.

### How this project does it

**`config.py:14`:** Default model is `deepseek-chat` via `https://api.deepseek.com`. The `llm_api_key` property (`config.py:46-47`) checks both `LLM_API_KEY` and `OPENAI_API_KEY` env vars.

**`llm_service.py:28-32`:**
```python
self.client = OpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)
```

The `base_url` parameter is the key — by changing it, you can switch between OpenAI, DeepSeek, or any self-hosted LLM (vLLM, Ollama, etc.) without code changes.

### What to learn

- The OpenAI Python client is the de facto standard. Most LLM providers mimic its API.
- Separating the LLM from the embedding model gives flexibility. This project can use DeepSeek for generation and a local BGE model for embeddings.
- Always configure through environment variables, never hard-code API keys or URLs.

---

## 9. Evaluation: Is Your RAG Actually Good?

**File:** `backend/tests/test_ragas.py`

### Concept

"How do I know my RAG is working?" is the hardest question in LLMOps. RAGAS (RAG Assessment) is a framework that evaluates RAG pipelines automatically.

### Metrics explained

| Metric | What it measures | Example |
|--------|-----------------|---------|
| **Context Precision** | Are retrieved documents actually relevant? | Question: "Port của Kafka?" → Retrieved doc about Kafka config = high precision. Retrieved doc about dogs = low. |
| **Context Recall** | Did we miss important documents in the DB? | DB has 3 relevant docs, we retrieved 2 = 67% recall. |
| **Faithfulness** | Does the answer stick to the documents, or hallucinate? | Doc says "Kafka uses port 9092", answer says "9092" = faithful. Answer says "8080" = hallucination. |
| **Answer Relevancy** | Does the answer address the question? | Question about ports, answer about installation = irrelevant. |

### How this project does it

**`test_ragas.py:33-57`:** Uses RAGAS to evaluate with an LLM-as-judge. RAGAS calls an LLM (requires `OPENAI_API_KEY`) to score each metric.

**`test_ragas.py:65-73`:** Test cases are defined as dictionaries with question, retrieved contexts, generated answer, and ground truth:
```python
{
    "question": "Tài liệu này nói về chủ đề gì?",
    "contexts": ["Tài liệu hướng dẫn sử dụng phần mềm quản lý dự án."],
    "answer": "...",
    "ground_truth": "...",
}
```

### What to learn

- Evaluation is not optional for production RAG. You need metrics to know if changes (new chunker, new model) actually help.
- LLM-as-judge (using an LLM to evaluate another LLM's output) is the current best practice. Human evaluation doesn't scale.
- A test suite of 50-100 curated question-answer pairs (ground truth dataset) is the minimum for detecting regressions.

---

## 10. Production Patterns in This Codebase

### Graceful Degradation

Three components can fail without bringing down the system:
- **Reranker** (`reranker.py:59-61`): missing model → pass-through
- **Query rewriter** (`query_rewriter.py:49-51`): API failure → use original query
- **Health check** (`health.py`): any component down → returns "degraded" instead of 500

### Lazy Initialization with Thread Safety

Both `EmbedderService` and `RerankerService` use this pattern:
```python
def _ensure_model(self):
    if self._model_loaded:
        return
    with self._lock:
        if self._model_loaded:  # Check again inside lock
            return
        # Load model...
        self._model_loaded = True
```
This is double-checked locking. It avoids loading heavy models at import time and prevents race conditions in multi-worker setups.

### Singleton Services

`VectorStore`, `ChatMemory`, and `RerankerService` override `__new__` to ensure only one instance exists. This is important because models consume GBs of RAM — loading them twice would OOM.

---

## How to Experiment With This Project

1. **Change the chunk size** — Edit `CHUNK_SIZE` in `.env`. Compare RAGAS scores at 256, 512, 1024. See how it affects retrieval quality.
2. **Try different embedding models** — Set `EMBEDDING_MODEL=intfloat/multilingual-e5-large` or `keepitreal/vietnamese-sbert`. Compare Vietnamese retrieval quality.
3. **Switch LLMs** — Point `LLM_BASE_URL` to a local Ollama instance. See how the answer style changes.
4. **Tune hybrid search weights** — Set `VECTOR_WEIGHT=0.5` and `KEYWORD_WEIGHT=0.5`. Does it improve precision for code-heavy documents?
5. **Add a proper BM25 index** — Replace the simple token scorer in `vector_store.py:140-166` with a real BM25 implementation. This is the most impactful improvement for mixed-language content.
6. **Build a ground truth dataset** — Write 50 question-answer pairs for your own documents. Run `test_ragas.py` before and after each change.

---

## Key Files to Read First

| Order | File | Why |
|-------|------|-----|
| 1 | `routers/chat.py` | See the full RAG pipeline orchestrated in 50 lines |
| 2 | `services/retriever.py` | Understand hybrid search merging |
| 3 | `services/chunker.py` | Learn semantic chunking + parent-child pattern |
| 4 | `services/embedder.py` | See model loading and embedding generation |
| 5 | `services/llm_service.py` | Understand prompt construction |
| 6 | `services/reranker.py` | Cross-encoder re-ranking |
| 7 | `services/query_rewriter.py` | Multi-turn conversation handling |
