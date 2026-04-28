import time
import logging
from fastapi import APIRouter, HTTPException, Depends
from app.models.schemas import ChatRequest, ChatResponse, SourceDocument
from app.services.query_rewriter import query_rewriter
from app.services.retriever import hybrid_search
from app.services.reranker import reranker_service
from app.services.llm_service import llm_service
from app.services.chat_memory import chat_memory
from app.utils.auth import require_api_key

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_api_key)])


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()

    try:
        history_messages = chat_memory.get_history_messages(request.session_id)
        rewritten_query = query_rewriter.rewrite_query(request.query, history_messages)

        search_results = hybrid_search(
            query=rewritten_query,
            top_k=20,
        )

        reranked_results = reranker_service.rerank(
            query=rewritten_query,
            documents=search_results,
            top_k=request.top_k,
        )

        context_docs = [
            {
                "content": doc[0].get("content", ""),
                "parent_content": doc[0].get("parent_content", ""),
                "file_name": doc[0].get("file_name", ""),
                "metadata": doc[0].get("metadata", {}),
            }
            for doc in reranked_results
        ]

        history_formatted = chat_memory.get_history_formatted(request.session_id)

        answer = llm_service.generate_response(
            question=request.query,
            context_docs=context_docs,
            chat_history=history_formatted,
        )

        chat_memory.add_message(request.session_id, "user", request.query)
        chat_memory.add_message(request.session_id, "assistant", answer)

        sources = [
            SourceDocument(
                content=doc["content"][:500],
                metadata={
                    "file_name": doc["file_name"],
                    **doc.get("metadata", {}),
                },
                relevance_score=reranked_results[i][1] if i < len(reranked_results) else 0.0,
            )
            for i, doc in enumerate(context_docs)
        ]

        processing_time = (time.time() - start_time) * 1000

        return ChatResponse(
            session_id=request.session_id,
            answer=answer,
            sources=sources,
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
async def get_chat_history(session_id: str):
    history = chat_memory.get_history(session_id)
    return {"session_id": session_id, "history": history}


@router.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    chat_memory.clear_history(session_id)
    return {"session_id": session_id, "status": "cleared"}
