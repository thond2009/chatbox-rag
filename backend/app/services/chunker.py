import re
import logging
from typing import List, Tuple, Optional
from app.models.document import DocumentChunk, LoadedDocument
from app.config import settings
from app.utils.preprocessing import preprocess_text

logger = logging.getLogger(__name__)


def _split_sentences(text: str) -> List[str]:
    try:
        import spacy
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        doc = nlp(text[:100_000])
        return [sent.text.strip() for sent in doc.sents]
    except (ImportError, OSError):
        pass

    try:
        import nltk
        nltk.data.find("tokenizers/punkt")
        return nltk.sent_tokenize(text)
    except (ImportError, LookupError):
        pass

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _build_chunks_from_sentences(
    sentences: List[str],
    chunk_size: int,
    chunk_overlap: int,
) -> List[DocumentChunk]:
    chunks = []
    current_chunk: List[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append(DocumentChunk(content=chunk_text))

            overlap_text = ""
            overlap_chunk = []
            overlap_length = 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) > chunk_overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_length += len(s)
            current_chunk = overlap_chunk
            current_length = overlap_length

        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append(DocumentChunk(content=chunk_text))

    return chunks


def _create_parent_content(full_text: str, child_chunk: str) -> str:
    chunk_start = full_text.find(child_chunk)
    if chunk_start == -1:
        return child_chunk

    parent_size = settings.PARENT_CHUNK_SIZE
    half_parent = parent_size // 2
    text_len = len(full_text)

    parent_start = max(0, chunk_start - half_parent)
    parent_end = min(text_len, chunk_start + len(child_chunk) + half_parent)

    parent = full_text[parent_start:parent_end]

    if parent_start > 0:
        parent = "..." + parent
    if parent_end < text_len:
        parent = parent + "..."

    return parent


def chunk_document(doc: LoadedDocument) -> LoadedDocument:
    logger.info(f"Chunking document: {doc.file_name}")
    text = doc.content
    sentences = _split_sentences(text)

    if not sentences:
        doc.chunks = [DocumentChunk(content=text, parent_content=text, metadata=doc.metadata)]
        return doc

    child_chunks = _build_chunks_from_sentences(
        sentences,
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )

    for chunk in child_chunks:
        chunk.parent_content = _create_parent_content(text, chunk.content)
        chunk.metadata = {
            **doc.metadata,
            "chunk_index": len(doc.chunks),
        }
        doc.chunks.append(chunk)

    logger.info(f"Created {len(doc.chunks)} chunks from '{doc.file_name}'")
    return doc
