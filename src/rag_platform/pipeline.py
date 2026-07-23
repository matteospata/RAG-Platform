from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .chunking import chunk_document
from .embeddings import EmbeddingModel
from .ingestion import load_documents
from .schemas import PipelineRun
from .store import SQLiteVectorStore

logger = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(self, store: SQLiteVectorStore, embeddings: EmbeddingModel, chunk_size: int, chunk_overlap: int) -> None:
        self.store = store
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def run(self, source_path: str | Path) -> PipelineRun:
        run = PipelineRun(run_id=uuid.uuid4().hex)
        documents = load_documents(source_path)
        run.documents_seen = len(documents)
        self.store.record_run(run.model_dump(mode="json"))
        try:
            for document in documents:
                if not self.store.upsert_document(document):
                    continue
                chunks = chunk_document(document, self.chunk_size, self.chunk_overlap)
                vectors = self.embeddings.embed([chunk.text for chunk in chunks])
                self.store.upsert_chunks(chunks, vectors)
                run.documents_indexed += 1
                run.chunks_indexed += len(chunks)
            run.status = "completed"
        except Exception as exc:
            logger.exception("Indexing run failed")
            run.status = "failed"
            run.error = str(exc)
        run.finished_at = datetime.now(timezone.utc)
        self.store.record_run(run.model_dump(mode="json"))
        if run.status == "failed":
            raise RuntimeError(run.error)
        return run

