from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.utils.logging_config import setup_logging
from app.routers import chat, ingestion, health

setup_logging()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url=f"{settings.API_V1_PREFIX}/docs",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix=settings.API_V1_PREFIX, tags=["Chat"])
app.include_router(ingestion.router, prefix=settings.API_V1_PREFIX, tags=["Ingestion"])
app.include_router(health.router, prefix=settings.API_V1_PREFIX, tags=["Health"])

import os
frontend_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


@app.get(f"{settings.API_V1_PREFIX}/info")
async def info():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "endpoints": {
            "chat": f"{settings.API_V1_PREFIX}/chat",
            "ingest": f"{settings.API_V1_PREFIX}/ingest",
            "health": f"{settings.API_V1_PREFIX}/health",
            "docs": f"{settings.API_V1_PREFIX}/docs",
        },
    }
