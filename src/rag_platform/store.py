from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .embeddings import cosine_similarity
from .schemas import Chunk, RetrievalResult, SourceDocument


class SQLiteVectorStore:
    """Small, inspectable vector store suitable for a portfolio project and local demos."""

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS documents (
              document_id TEXT PRIMARY KEY, source TEXT NOT NULL, title TEXT NOT NULL,
              text TEXT NOT NULL, content_hash TEXT NOT NULL, metadata_json TEXT NOT NULL,
              ingested_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL, source TEXT NOT NULL,
              title TEXT NOT NULL, text TEXT NOT NULL, chunk_index INTEGER NOT NULL,
              metadata_json TEXT NOT NULL, embedding_json TEXT NOT NULL,
              FOREIGN KEY(document_id) REFERENCES documents(document_id)
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
            CREATE TABLE IF NOT EXISTS pipeline_runs (
              run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT,
              documents_seen INTEGER NOT NULL, documents_indexed INTEGER NOT NULL,
              chunks_indexed INTEGER NOT NULL, status TEXT NOT NULL, error TEXT
            );
            """
        )
        self.connection.commit()

    def upsert_document(self, document: SourceDocument) -> bool:
        existing = self.connection.execute(
            "SELECT content_hash FROM documents WHERE document_id = ?", (document.document_id,)
        ).fetchone()
        if existing and existing["content_hash"] == document.content_hash:
            return False
        self.connection.execute(
            """INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(document_id) DO UPDATE SET source=excluded.source, title=excluded.title,
            text=excluded.text, content_hash=excluded.content_hash, metadata_json=excluded.metadata_json,
            ingested_at=excluded.ingested_at""",
            (document.document_id, document.source, document.title, document.text, document.content_hash,
             json.dumps(document.metadata), document.ingested_at.isoformat()),
        )
        self.connection.execute("DELETE FROM chunks WHERE document_id = ?", (document.document_id,))
        self.connection.commit()
        return True

    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.connection.executemany(
            """INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [(chunk.chunk_id, chunk.document_id, chunk.source, chunk.title, chunk.text, chunk.chunk_index,
              json.dumps(chunk.metadata), json.dumps(vector)) for chunk, vector in zip(chunks, embeddings)],
        )
        self.connection.commit()

    def search(self, query_embedding: list[float], top_k: int, min_score: float) -> list[RetrievalResult]:
        rows = self.connection.execute("SELECT * FROM chunks").fetchall()
        ranked = sorted(
            ((cosine_similarity(query_embedding, json.loads(row["embedding_json"])), row) for row in rows),
            key=lambda pair: pair[0], reverse=True,
        )
        return [
            RetrievalResult(
                chunk_id=row["chunk_id"], document_id=row["document_id"], source=row["source"],
                title=row["title"], text=row["text"], score=score,
                metadata=json.loads(row["metadata_json"]),
            )
            for score, row in ranked[:top_k] if score >= min_score
        ]

    def list_documents(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT document_id, source, title, content_hash, ingested_at FROM documents ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(row) for row in rows]

    def record_run(self, run: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO pipeline_runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run["run_id"], run["started_at"], run.get("finished_at"), run["documents_seen"],
             run["documents_indexed"], run["chunks_indexed"], run["status"], run.get("error")),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

