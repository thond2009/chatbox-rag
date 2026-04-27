import logging
from typing import Dict, List, Optional
from collections import defaultdict
from app.models.schemas import ChatMessage
from app.config import settings

logger = logging.getLogger(__name__)


class ChatMemory:
    _instance: Optional["ChatMemory"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._stores = defaultdict(list)
        return cls._instance

    def add_message(self, session_id: str, role: str, content: str):
        max_history = settings.MAX_CHAT_HISTORY * 2
        self._stores[session_id].append({"role": role, "content": content})
        if len(self._stores[session_id]) > max_history:
            self._stores[session_id] = self._stores[session_id][-max_history:]

    def get_history(self, session_id: str) -> List[dict]:
        return list(self._stores.get(session_id, []))

    def get_history_formatted(self, session_id: str) -> List[str]:
        history = self.get_history(session_id)
        return [f"{msg['role']}: {msg['content']}" for msg in history]

    def clear_history(self, session_id: str):
        self._stores.pop(session_id, None)

    def get_history_messages(self, session_id: str) -> List[ChatMessage]:
        history = self.get_history(session_id)
        return [ChatMessage(role=msg["role"], content=msg["content"]) for msg in history]


chat_memory = ChatMemory()
