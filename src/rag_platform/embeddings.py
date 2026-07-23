from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingModel(ABC):
    dimension: int
    name: str

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbeddingModel(EmbeddingModel):
    """Dependency-light baseline; deterministic and useful for demos/tests."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension
        self.name = f"hash-{dimension}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            tokens = text.lower().split()
            for token in tokens:
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 else -1.0
                vector[index] += sign
            norm = float(np.linalg.norm(vector))
            vectors.append((vector / norm if norm else vector).tolist())
        return vectors


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("Install the 'local' extra to use sentence_transformers") from exc
        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())
        self.name = model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key: str, model_name: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the 'llm' extra to use OpenAI") from exc
        self.client = OpenAI(api_key=api_key)
        self.name = model_name
        self.dimension = 1536 if "3-small" in model_name else 3072 if "3-large" in model_name else 1536

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.name, input=texts)
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]


def build_embedding_model(provider: str, model_name: str, dimension: int, api_key: str | None) -> EmbeddingModel:
    if provider == "hash":
        return HashEmbeddingModel(dimension)
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddingModel(model_name)
    if provider == "openai":
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for the openai embedding provider")
        return OpenAIEmbeddingModel(api_key, model_name)
    raise ValueError(f"Unknown embedding provider: {provider}")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_array, right_array = np.asarray(left), np.asarray(right)
    denominator = math.sqrt(float(left_array @ left_array) * float(right_array @ right_array))
    return float(left_array @ right_array / denominator) if denominator else 0.0

