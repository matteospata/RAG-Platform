from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, get_settings
from .embeddings import EmbeddingModel, build_embedding_model
from .llm import build_chat_model
from .rag import RAGService
from .store import SQLiteVectorStore


@dataclass
class Container:
    settings: Settings
    store: SQLiteVectorStore
    embeddings: EmbeddingModel
    rag: RAGService


def build_container(settings: Settings | None = None) -> Container:
    current = settings or get_settings()
    current.ensure_directories()
    store = SQLiteVectorStore(current.db_path)
    embeddings = build_embedding_model(
        current.embedding_provider, current.embedding_model, current.embedding_dimension, current.openai_api_key
    )
    chat = build_chat_model(current.llm_provider, current.llm_model, current.openai_api_key)
    return Container(current, store, embeddings, RAGService(store, embeddings, chat, current.top_k, current.min_score))
