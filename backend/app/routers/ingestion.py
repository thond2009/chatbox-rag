import os
import logging
import uuid
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from typing import Optional
from app.config import settings
from app.services.document_loader import load_document, detect_file_type
from app.services.chunker import chunk_document
from app.services.vector_store import vector_store
from app.models.schemas import IngestionResponse, ListDocumentsResponse, DeleteDocumentRequest
from app.utils.auth import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    file: UploadFile = File(...),
    file_type: Optional[str] = Form("auto"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    if file_type == "auto":
        file_type = detect_file_type(file.filename or "unknown.txt")

    file_ext = ".txt"
    if file_type == "pdf":
        file_ext = ".pdf"
    elif file_type == "md":
        file_ext = ".md"
    elif file_type == "html":
        file_ext = ".html"

    file_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4().hex}{file_ext}")

    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        doc = load_document(file_path, file_type)
        doc = chunk_document(doc)

        if not doc.chunks:
            return IngestionResponse(
                status="warning",
                document_id=doc.id,
                chunks_count=0,
                file_name=doc.file_name,
                message="Document was loaded but could not be chunked: content is empty",
            )

        chunks_count = vector_store.upsert_chunks(doc.id, doc.chunks)

        return IngestionResponse(
            status="success",
            document_id=doc.id,
            chunks_count=chunks_count,
            file_name=doc.file_name,
            message=f"Successfully processed '{doc.file_name}': {chunks_count} chunks ingested",
        )

    except Exception as e:
        logger.error(f"Ingestion error: {e}", exc_info=True)
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents", response_model=ListDocumentsResponse)
async def list_documents():
    try:
        docs = vector_store.list_documents()
        return ListDocumentsResponse(documents=docs, total=len(docs))
    except Exception as e:
        logger.error(f"List documents error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    try:
        vector_store.delete_by_document_id(document_id)
        return {"status": "deleted", "document_id": document_id}
    except Exception as e:
        logger.error(f"Delete document error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/text", response_model=IngestionResponse)
async def ingest_text(
    content: str = Form(..., max_length=1_000_000),
    file_name: str = Form("manual_input.txt"),
    file_type: str = Form("txt"),
):
    if not content.strip():
        raise HTTPException(status_code=400, detail="Content must not be empty")
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(settings.UPLOAD_DIR, f"{uuid.uuid4().hex}.txt")

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        doc = load_document(file_path, file_type)
        doc = chunk_document(doc)

        if not doc.chunks:
            return IngestionResponse(
                status="warning",
                document_id=doc.id,
                chunks_count=0,
                file_name=doc.file_name,
                message="Empty content",
            )

        chunks_count = vector_store.upsert_chunks(doc.id, doc.chunks)

        return IngestionResponse(
            status="success",
            document_id=doc.id,
            chunks_count=chunks_count,
            file_name=doc.file_name,
            message=f"Successfully ingested text as '{file_name}': {chunks_count} chunks",
        )

    except Exception as e:
        logger.error(f"Text ingestion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
