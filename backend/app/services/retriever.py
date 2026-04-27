import logging
from typing import List, Tuple, Dict, Optional
from app.config import settings
from app.services.embedder import embedder_service
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


def hybrid_search(
    query: str,
    top_k: int = None,
    filter_dict: Optional[Dict] = None,
) -> List[Tuple[Dict, float]]:
    if top_k is None:
        top_k = settings.HYBRID_SEARCH_TOP_K

    logger.info(f"Hybrid search for: '{query[:100]}...'")

    query_embedding = embedder_service.embed_text(query)

    vector_results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k * 2,
        filter_dict=filter_dict,
    )

    keyword_results = vector_store.keyword_search(
        query=query,
        top_k=top_k * 2,
        filter_dict=filter_dict,
    )

    vector_weight = settings.VECTOR_WEIGHT
    keyword_weight = settings.KEYWORD_WEIGHT

    merged_scores: Dict[str, Tuple[Dict, float]] = {}

    for payload, score in vector_results:
        chunk_id = payload.get("chunk_id", str(id(payload)))
        merged_scores[chunk_id] = (payload, score * vector_weight)

    for payload, score in keyword_results:
        chunk_id = payload.get("chunk_id", str(id(payload)))
        if chunk_id in merged_scores:
            merged_scores[chunk_id] = (
                payload,
                merged_scores[chunk_id][1] + score * keyword_weight,
            )
        else:
            merged_scores[chunk_id] = (payload, score * keyword_weight)

    sorted_results = sorted(merged_scores.values(), key=lambda x: x[1], reverse=True)
    return sorted_results[:top_k]
