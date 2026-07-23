from pathlib import Path

from rag_platform.embeddings import HashEmbeddingModel
from rag_platform.pipeline import IndexingPipeline
from rag_platform.store import SQLiteVectorStore


def test_pipeline_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text("Python is used for data engineering and machine learning.", encoding="utf-8")
    store = SQLiteVectorStore(tmp_path / "test.db")
    pipeline = IndexingPipeline(store, HashEmbeddingModel(64), chunk_size=200, chunk_overlap=20)

    first = pipeline.run(source)
    second = pipeline.run(source)

    assert first.documents_indexed == 1
    assert second.documents_indexed == 0
    assert len(store.list_documents()) == 1
    assert store.search(HashEmbeddingModel(64).embed(["data engineering"])[0], 3, -1)
    store.close()
