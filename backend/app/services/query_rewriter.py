import logging
from typing import List, Optional
from openai import OpenAI
from app.config import settings
from app.models.schemas import ChatMessage

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.LLM_MODEL

    def rewrite_query(
        self,
        current_query: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> str:
        if not chat_history or len(chat_history) == 0:
            return current_query

        history_text = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in chat_history[-6:]]
        )

        prompt = f"""Bạn là trợ lý viết lại câu hỏi. Nhiệm vụ của bạn là viết lại câu hỏi hiện tại thành một câu hỏi độc lập, có đầy đủ ngữ cảnh dựa trên lịch sử trò chuyện.

LỊCH SỬ TRÒ CHUYỆN:
{history_text}

CÂU HỎI HIỆN TẠI: {current_query}

CÂU HỎI ĐỘC LẬP (chỉ trả về câu hỏi, không giải thích):"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"Query rewritten: '{current_query[:50]}...' -> '{rewritten[:50]}...'")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewriting failed: {e}, using original query")
            return current_query


query_rewriter = QueryRewriter()
