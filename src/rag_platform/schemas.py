from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SourceDocument(BaseModel):
    document_id: str
    source: str
    title: str
    text: str
    content_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    ingested_at: datetime = Field(default_factory=utc_now)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    chunk_index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[RetrievalResult] = Field(default_factory=list)
    model: str
    latency_ms: float


class PipelineRun(BaseModel):
    run_id: str
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    documents_seen: int = 0
    documents_indexed: int = 0
    chunks_indexed: int = 0
    status: str = "running"
    error: str | None = None

