from dataclasses import dataclass, field
from typing import List, Dict, Optional
from uuid import UUID, uuid4


@dataclass
class DocumentChunk:
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    parent_content: str = ""
    metadata: Dict = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class LoadedDocument:
    id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    file_name: str = ""
    file_type: str = ""
    metadata: Dict = field(default_factory=dict)
    chunks: List[DocumentChunk] = field(default_factory=list)
