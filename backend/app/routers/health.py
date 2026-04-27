from datetime import datetime, timezone
from fastapi import APIRouter
from app.config import settings
from app.services.vector_store import vector_store
from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    components = {}
    try:
        qdrant_info = vector_store.get_collection_info()
        components["qdrant"] = {"status": "connected", **qdrant_info}
    except Exception:
        components["qdrant"] = {"status": "disconnected"}

    try:
        from app.services.embedder import embedder_service
        _ = embedder_service.dimension
        components["embedder"] = {"status": "configured", "dimension": embedder_service.dimension}
    except Exception:
        components["embedder"] = {"status": "not configured"}

    try:
        from app.services.reranker import reranker_service
        if reranker_service.model is not None:
            components["reranker"] = {"status": "loaded", "model": settings.RERANKER_MODEL}
        else:
            components["reranker"] = {"status": "not loaded", "model": settings.RERANKER_MODEL}
    except Exception:
        components["reranker"] = {"status": "not available"}

    all_healthy = all(
        c.get("status") in ("connected", "configured", "loaded")
        for c in components.values()
    )

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        components=components,
    )
