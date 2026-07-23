from pathlib import Path

from rag_platform.embeddings import HashEmbeddingModel
from rag_platform.llm import ExtractiveChatModel
from rag_platform.rag import RAGService
from rag_platform.store import SQLiteVectorStore
from rag_platform.pipeline import IndexingPipeline


def test_rag_returns_citations(tmp_path: Path) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text("The pipeline uses chunking, embeddings, and retrieval.", encoding="utf-8")
    store = SQLiteVectorStore(tmp_path / "test.db")
    embeddings = HashEmbeddingModel(64)
    IndexingPipeline(store, embeddings, 200, 20).run(source)
    result = RAGService(store, embeddings, ExtractiveChatModel(), top_k=3, min_score=-1).ask("embedding")
    assert result.citations
    assert result.model == "extractive-local"
    store.close()
