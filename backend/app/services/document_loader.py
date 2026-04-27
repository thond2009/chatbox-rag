import os
import re
from typing import Optional
from app.models.document import LoadedDocument
from app.utils.preprocessing import preprocess_text, extract_metadata_from_content
from app.config import settings
import logging

logger = logging.getLogger(__name__)


def detect_file_type(file_name: str) -> str:
    ext = os.path.splitext(file_name)[1].lower()
    mapping = {
        ".pdf": "pdf",
        ".md": "md",
        ".markdown": "md",
        ".html": "html",
        ".htm": "html",
        ".txt": "txt",
    }
    return mapping.get(ext, "txt")


def load_pdf(file_path: str) -> str:
    try:
        import fitz
        doc = fitz.open(file_path)
        text_parts = []
        for page_num, page in enumerate(doc, 1):
            page_text = page.get_text("text")
            if page_text.strip():
                text_parts.append(f"[Page {page_num}]\n{page_text}")
        doc.close()
        return "\n\n".join(text_parts)
    except ImportError:
        raise ImportError("PyMuPDF (fitz) is required for PDF loading. Install with: pip install PyMuPDF")


def load_markdown(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_html(file_path: str) -> str:
    try:
        from bs4 import BeautifulSoup
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)
    except ImportError:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        content = re.sub(r"<[^>]+>", " ", content)
        return content


def load_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


LOADER_MAP = {
    "pdf": load_pdf,
    "md": load_markdown,
    "html": load_html,
    "txt": load_text,
}


def load_document(file_path: str, file_type: Optional[str] = None) -> LoadedDocument:
    file_name = os.path.basename(file_path)
    if file_type == "auto" or file_type is None:
        file_type = detect_file_type(file_name)

    loader = LOADER_MAP.get(file_type)
    if loader is None:
        raise ValueError(f"Unsupported file type: {file_type}")

    logger.info(f"Loading document: {file_name} (type: {file_type})")
    raw_content = loader(file_path)
    cleaned_content = preprocess_text(raw_content)

    metadata = extract_metadata_from_content(cleaned_content)
    metadata.update({
        "file_name": file_name,
        "file_type": file_type,
        "file_path": file_path,
    })

    doc = LoadedDocument(
        content=cleaned_content,
        file_name=file_name,
        file_type=file_type,
        metadata=metadata,
    )
    logger.info(f"Loaded document '{file_name}': {len(cleaned_content)} chars")
    return doc
