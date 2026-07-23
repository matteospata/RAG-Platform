from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator


class Settings(BaseModel):
    """Typed configuration; environment variables are deliberately explicit."""

    env: str = Field(default="local", alias="RAG_ENV")
    data_dir: Path = Field(default=Path("./data"), alias="RAG_DATA_DIR")
    db_path: Path = Field(default=Path("./data/rag.db"), alias="RAG_DB_PATH")
    log_level: str = Field(default="INFO", alias="RAG_LOG_LEVEL")
    embedding_provider: str = Field(default="hash", alias="RAG_EMBEDDING_PROVIDER")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="RAG_EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=384, alias="RAG_EMBEDDING_DIMENSION", ge=8)
    llm_provider: str = Field(default="extractive", alias="RAG_LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o-mini", alias="RAG_LLM_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    top_k: int = Field(default=5, alias="RAG_TOP_K", ge=1, le=50)
    min_score: float = Field(default=0.15, alias="RAG_MIN_SCORE", ge=-1, le=1)
    chunk_size: int = Field(default=800, alias="RAG_CHUNK_SIZE", ge=100)
    chunk_overlap: int = Field(default=120, alias="RAG_CHUNK_OVERLAP", ge=0)

    @field_validator("embedding_provider", "llm_provider", mode="before")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return str(value).strip().lower()

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv()
    values = {field.alias: os.getenv(field.alias) for field in Settings.model_fields.values() if os.getenv(field.alias)}
    settings = Settings.model_validate(values)
    settings.ensure_directories()
    return settings

