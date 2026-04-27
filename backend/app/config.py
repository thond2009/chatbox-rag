import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    PROJECT_NAME: str = "RAG Chatbot System"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"

    LLM_API_KEY: Optional[str] = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: Optional[str] = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))

    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    EMBEDDING_API_KEY: Optional[str] = os.getenv("EMBEDDING_API_KEY", "")
    EMBEDDING_BASE_URL: Optional[str] = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1")

    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "rag_documents")

    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "5"))

    HYBRID_SEARCH_TOP_K: int = int(os.getenv("HYBRID_SEARCH_TOP_K", "20"))
    VECTOR_WEIGHT: float = float(os.getenv("VECTOR_WEIGHT", "0.7"))
    KEYWORD_WEIGHT: float = float(os.getenv("KEYWORD_WEIGHT", "0.3"))

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    PARENT_CHUNK_SIZE: int = int(os.getenv("PARENT_CHUNK_SIZE", "1024"))

    MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "10"))

    CORS_ORIGINS: list = ["*"]

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")

    @property
    def llm_api_key(self) -> str:
        return self.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")

    @property
    def llm_base_url(self) -> str:
        return self.LLM_BASE_URL or os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
