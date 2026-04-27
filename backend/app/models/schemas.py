from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from uuid import UUID, uuid4


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(default_factory=lambda: str(uuid4()))
    query: str = Field(..., min_length=1, description="User's question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of documents to retrieve")


class SourceDocument(BaseModel):
    content: str = Field(..., description="Document chunk content")
    metadata: Dict = Field(default_factory=dict)
    relevance_score: float = Field(default=0.0)


class ChatResponse(BaseModel):
    session_id: str
    answer: str = Field(..., description="LLM generated answer")
    sources: List[SourceDocument] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)


class IngestionRequest(BaseModel):
    file_type: str = Field(default="auto", description="txt, pdf, md, html, or auto-detect")
    metadata: Optional[Dict] = Field(default_factory=dict)


class IngestionResponse(BaseModel):
    status: str
    document_id: str
    chunks_count: int
    file_name: str
    message: str


class ListDocumentsResponse(BaseModel):
    documents: List[Dict]
    total: int


class DeleteDocumentRequest(BaseModel):
    document_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    components: Dict
