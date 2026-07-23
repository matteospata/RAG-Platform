from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .container import Container, build_container
from .logging_config import configure_logging
from .pipeline import IndexingPipeline

logger = logging.getLogger(__name__)
container: Container | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2_000)


class IngestRequest(BaseModel):
    path: str = Field(default="data/knowledge_base", min_length=1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global container
    container = build_container()
    configure_logging(container.settings.log_level)
    yield
    if container:
        container.store.close()


app = FastAPI(title="RAG Data Platform", version="0.1.0", lifespan=lifespan)


def get_container() -> Container:
    if container is None:
        raise HTTPException(status_code=503, detail="Application is not initialized")
    return container


@app.get("/health")
def health() -> dict:
    current = get_container()
    return {"status": "ok", "embedding_model": current.embeddings.name, "llm_model": current.rag.chat.name}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    current = get_container()
    source = Path(request.path)
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {source}")
    run = IndexingPipeline(current.store, current.embeddings, current.settings.chunk_size, current.settings.chunk_overlap).run(source)
    return run.model_dump(mode="json")


@app.post("/query")
def query(request: QueryRequest) -> dict:
    try:
        return get_container().rag.ask(request.question).model_dump(mode="json")
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/documents")
def documents() -> list[dict]:
    return get_container().store.list_documents()

