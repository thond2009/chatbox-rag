import logging
from typing import List, Optional
from openai import OpenAI
from app.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Bạn là chuyên gia kỹ thuật hỗ trợ dự án. Nhiệm vụ của bạn là trả lời câu hỏi dựa TRÊN CÁC TÀI LIỆU được cung cấp dưới đây.

RÀNG BUỘC:
1. NẾU thông tin KHÔNG có trong tài liệu, hãy trả lời chính xác: "Dữ liệu hiện tại không chứa thông tin này." Không tự suy diễn.
2. Trích dẫn nguồn (tên file hoặc số trang) sau mỗi luận điểm nếu có.
3. Trình bày bằng Markdown, sử dụng bullet points để dễ đọc.

---
TÀI LIỆU TRUY XUẤT:
{context}

---
LỊCH SỬ TRÒ CHUYỆN:
{chat_history}

---
CÂU HỎI HIỆN TẠI: {question}"""


class LLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
        self.model = settings.LLM_MODEL

    def generate_response(
        self,
        question: str,
        context_docs: List[dict],
        chat_history: Optional[List[str]] = None,
    ) -> str:
        context_parts = []
        for i, doc in enumerate(context_docs):
            content = doc.get("parent_content", doc.get("content", ""))
            source = doc.get("file_name", f"Source {i+1}")
            context_parts.append(f"[{i+1}] Nguồn: {source}\n{content}")

        context = "\n\n---\n\n".join(context_parts)

        if chat_history:
            history_text = "\n".join(chat_history[-10:])
        else:
            history_text = "Không có lịch sử trò chuyện."

        prompt = SYSTEM_PROMPT.format(
            context=context,
            chat_history=history_text,
            question=question,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
            )
            answer = response.choices[0].message.content
            logger.info(f"LLM response generated: {len(answer)} chars")
            return answer
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise


llm_service = LLMService()
