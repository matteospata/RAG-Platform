from __future__ import annotations

import re

from .schemas import Chunk, SourceDocument


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if part.strip()]


def chunk_document(document: SourceDocument, chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Create deterministic, sentence-aware chunks without splitting words."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    sentences = _split_sentences(document.text)
    chunks: list[Chunk] = []
    current: list[str] = []
    current_length = 0
    index = 0
    for sentence in sentences:
        if current and current_length + len(sentence) + 1 > chunk_size:
            text = " ".join(current).strip()
            chunks.append(_make_chunk(document, index, text))
            index += 1
            tail: list[str] = []
            tail_length = 0
            for previous in reversed(current):
                if tail_length + len(previous) + 1 > overlap:
                    break
                tail.insert(0, previous)
                tail_length += len(previous) + 1
            current, current_length = tail, tail_length
        current.append(sentence)
        current_length += len(sentence) + 1
    if current:
        chunks.append(_make_chunk(document, index, " ".join(current).strip()))
    return chunks


def _make_chunk(document: SourceDocument, index: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{document.document_id}:{index}",
        document_id=document.document_id,
        source=document.source,
        title=document.title,
        text=text,
        chunk_index=index,
        metadata=document.metadata,
    )

