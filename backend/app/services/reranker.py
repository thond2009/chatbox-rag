import logging
import threading
from typing import List, Tuple, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from app.config import settings

logger = logging.getLogger(__name__)


class RerankerService:
    _instance: Optional["RerankerService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.model_name = settings.RERANKER_MODEL
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._model_loaded = False
        self._lock = threading.Lock()
        self._initialized = True

    def _ensure_model(self):
        if self._model_loaded:
            return
        with self._lock:
            if self._model_loaded:
                return
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.model = self.model.cuda()
                self.model.eval()
                logger.info(f"Loaded reranker model: {self.model_name} on {self.device}")
            except Exception as e:
                logger.warning(f"Could not load reranker model: {e}. Reranking disabled.")
            self._model_loaded = True

    def rerank(
        self,
        query: str,
        documents: List[Tuple[Dict, float]],
        top_k: Optional[int] = None,
    ) -> List[Tuple[Dict, float]]:
        self._ensure_model()
        if top_k is None:
            top_k = settings.RERANKER_TOP_K

        if self.model is None or self.tokenizer is None:
            logger.warning("Reranker not available, returning top results as-is")
            return documents[:top_k]

        if not documents:
            return []

        contents = [doc[0].get("content", "")[:512] for doc in documents]

        pairs = [[query, content] for content in contents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()

        scored_docs = list(zip(documents, scores.tolist()))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return [(doc[0], float(score)) for doc, score in scored_docs[:top_k]]


reranker_service = RerankerService()
