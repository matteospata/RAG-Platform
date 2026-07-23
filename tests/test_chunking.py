from rag_platform.chunking import chunk_document
from rag_platform.schemas import SourceDocument


def test_chunking_preserves_document_identity() -> None:
    document = SourceDocument(
        document_id="doc-1", source="test.md", title="Test", text="First sentence. Second sentence. Third sentence.", content_hash="hash"
    )
    chunks = chunk_document(document, chunk_size=25, overlap=5)
    assert chunks
    assert all(chunk.document_id == "doc-1" for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    document = SourceDocument(document_id="d", source="s", title="t", text="text", content_hash="h")
    try:
        chunk_document(document, chunk_size=10, overlap=10)
    except ValueError as exc:
        assert "overlap" in str(exc)
    else:
        raise AssertionError("Expected a ValueError")
