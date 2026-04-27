import logging
from typing import List, Optional
from app.config import settings

logger = logging.getLogger(__name__)


class EmbedderService:
    def __init__(self):
        self.provider = settings.EMBEDDING_PROVIDER
        self._model = None
        self._dimension = settings.EMBEDDING_DIM
        self._api_client = None

        if self.provider == "local":
            self._init_local()
        elif self.provider == "api":
            self._init_api()
        else:
            raise ValueError(f"Unknown embedding provider: {self.provider}. Use 'local' or 'api'.")

    def _init_local(self):
        try:
            from sentence_transformers import SentenceTransformer
            model_name = settings.EMBEDDING_MODEL
            logger.info(f"Loading local embedding model: {model_name}")
            self._model = SentenceTransformer(model_name, trust_remote_code=True)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Local embedding model loaded. Dimension: {self._dimension}")
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

    def _init_api(self):
        try:
            from openai import OpenAI
            api_key = settings.EMBEDDING_API_KEY or settings.llm_api_key
            base_url = settings.EMBEDDING_BASE_URL
            self._api_client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"Using API embedding: {settings.EMBEDDING_MODEL} via {base_url}")
        except ImportError:
            raise ImportError("openai is required for API embeddings. Install with: pip install openai")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if self.provider == "local":
            return self._embed_local(texts)
        else:
            return self._embed_api(texts)

    def _embed_local(self, texts: List[str]) -> List[List[float]]:
        texts = [t[:8000] for t in texts]
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _embed_api(self, texts: List[str]) -> List[List[float]]:
        texts = [t[:8000] for t in texts]
        response = self._api_client.embeddings.create(
            model=settings.EMBEDDING_MODEL,
            input=texts,
        )
        return [r.embedding for r in response.data]

    def embed_text(self, text: str) -> List[float]:
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []


embedder_service = EmbedderService()
