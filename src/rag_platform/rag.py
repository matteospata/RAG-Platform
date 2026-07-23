from __future__ import annotations

import time

from .embeddings import EmbeddingModel
from .llm import ChatModel
from .schemas import Answer, RetrievalResult
from .store import SQLiteVectorStore


class RAGService:
    def __init__(self, store: SQLiteVectorStore, embeddings: EmbeddingModel, chat: ChatModel,
                 top_k: int = 5, min_score: float = 0.15) -> None:
        self.store = store
        self.embeddings = embeddings
        self.chat = chat
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str) -> list[RetrievalResult]:
        query_vector = self.embeddings.embed([question])[0]
        return self.store.search(query_vector, self.top_k, self.min_score)

    @staticmethod
    def _context(results: list[RetrievalResult]) -> str:
        return "\n\n".join(
            f"[S{index}] {result.title} | {result.source}\n{result.text}"
            for index, result in enumerate(results, 1)
        )

    def ask(self, question: str) -> Answer:
        started = time.perf_counter()
        results = self.retrieve(question)
        answer = self.chat.answer(question, self._context(results))
        return Answer(
            question=question,
            answer=answer,
            citations=results,
            model=self.chat.name,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

