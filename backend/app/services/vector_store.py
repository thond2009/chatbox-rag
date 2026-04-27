import uuid
import time
import logging
from typing import List, Optional, Dict, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.config import settings
from app.models.document import DocumentChunk
from app.services.embedder import embedder_service

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_BASE_DELAY = 2


class VectorStore:
    _instance: Optional["VectorStore"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.qdrant_url = settings.QDRANT_URL
        self.collection_name = settings.QDRANT_COLLECTION
        self.client = None
        self._initialized = True

    def _ensure_connected(self):
        if self.client is not None:
            return

        for attempt in range(MAX_RETRIES):
            try:
                self.client = QdrantClient(url=self.qdrant_url)
                self._ensure_collection()
                return
            except Exception as e:
                self.client = None
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BASE_DELAY ** (attempt + 1)
                    logger.warning(
                        f"Qdrant connection failed (attempt {attempt+1}/{MAX_RETRIES}), "
                        f"retrying in {wait}s: {e}"
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"Failed to connect to Qdrant after {MAX_RETRIES} attempts")
                    raise

    def _ensure_collection(self):
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]
        if self.collection_name not in collection_names:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedder_service.dimension,
                    distance=Distance.COSINE,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="document_id",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="file_name",
                field_schema=qdrant_models.PayloadSchemaType.KEYWORD,
            )

    def upsert_chunks(self, document_id: str, chunks: List[DocumentChunk]) -> int:
        self._ensure_connected()
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = embedder_service.embed_texts(texts)

        points = []
        for i, chunk in enumerate(chunks):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={
                    "document_id": document_id,
                    "chunk_id": chunk.id,
                    "content": chunk.content,
                    "parent_content": chunk.parent_content,
                    "metadata": chunk.metadata,
                    "file_name": chunk.metadata.get("file_name", ""),
                    "file_type": chunk.metadata.get("file_type", ""),
                },
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        logger.info(f"Upserted {len(points)} chunks for document '{document_id}'")
        return len(points)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[Dict, float]]:
        self._ensure_connected()
        query_filter = None
        if filter_dict:
            must_conditions = []
            for key, value in filter_dict.items():
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            if must_conditions:
                query_filter = Filter(must=must_conditions)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            (hit.payload, hit.score)
            for hit in response.points
            if hit.payload is not None
        ]

    def keyword_search(
        self,
        query: str,
        top_k: int = 20,
        filter_dict: Optional[Dict] = None,
    ) -> List[Tuple[Dict, float]]:
        self._ensure_connected()
        query_tokens = query.lower().split()
        all_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=1000,
            with_payload=True,
        )

        scored = []
        for point in all_points:
            if point.payload is None:
                continue
            content = point.payload.get("content", "").lower()
            score = sum(1 for token in query_tokens if token in content)
            if score > 0:
                score = score / max(len(query_tokens), 1)
                scored.append((point.payload, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def delete_by_document_id(self, document_id: str) -> bool:
        self._ensure_connected()
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
            ),
        )
        logger.info(f"Deleted chunks for document '{document_id}'")
        return True

    def list_documents(self) -> List[dict]:
        self._ensure_connected()
        all_points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
        )
        docs_map = {}
        for point in all_points:
            if point.payload is None:
                continue
            doc_id = point.payload.get("document_id", "")
            if doc_id not in docs_map:
                docs_map[doc_id] = {
                    "document_id": doc_id,
                    "file_name": point.payload.get("file_name", ""),
                    "file_type": point.payload.get("file_type", ""),
                    "chunks_count": 0,
                }
            docs_map[doc_id]["chunks_count"] += 1

        return list(docs_map.values())

    def get_collection_info(self) -> dict:
        self._ensure_connected()
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
        }


vector_store = VectorStore()
