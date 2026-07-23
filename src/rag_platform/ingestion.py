from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Iterator

from .schemas import SourceDocument

logger = logging.getLogger(__name__)
SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".json", ".jsonl", ".csv"}


def _stable_id(source: str, content_hash: str) -> str:
    return hashlib.sha256(f"{source}:{content_hash}".encode()).hexdigest()[:24]


def _document(source: str, title: str, text: str, metadata: dict | None = None) -> SourceDocument:
    normalized = "\n".join(line.rstrip() for line in text.replace("\r\n", "\n").splitlines()).strip()
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return SourceDocument(
        document_id=_stable_id(source, content_hash),
        source=source,
        title=title or Path(source).stem,
        text=normalized,
        content_hash=content_hash,
        metadata=metadata or {},
    )


def _read_file(path: Path) -> Iterator[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        logger.debug("Skipping unsupported file: %s", path)
        return
    source = str(path)
    if suffix in {".md", ".markdown", ".txt"}:
        yield _document(source, path.stem.replace("_", " ").title(), path.read_text(encoding="utf-8"), {"file": source})
        return
    if suffix == ".jsonl":
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = str(record.get("text", record.get("content", "")))
            if text.strip():
                yield _document(
                    f"{source}#L{line_number}",
                    str(record.get("title", f"{path.stem} {line_number}")),
                    text,
                    {"file": source, "line": line_number, **record.get("metadata", {})},
                )
        return
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]
        for index, record in enumerate(records):
            if isinstance(record, str):
                text, title, metadata = record, f"{path.stem} {index}", {}
            else:
                text = str(record.get("text", record.get("content", "")))
                title = str(record.get("title", f"{path.stem} {index}"))
                metadata = record.get("metadata", {})
            if text.strip():
                yield _document(f"{source}#{index}", title, text, {"file": source, **metadata})
        return
    with path.open(newline="", encoding="utf-8") as handle:
        for row_number, row in enumerate(csv.DictReader(handle), 2):
            text = str(row.get("text", row.get("content", "")))
            if text.strip():
                title = str(row.get("title", f"{path.stem} {row_number}"))
                metadata = {k: v for k, v in row.items() if k not in {"text", "content", "title"} and v}
                yield _document(f"{source}#row{row_number}", title, text, {"file": source, **metadata})


def load_documents(path: str | Path) -> list[SourceDocument]:
    """Load supported documents recursively with stable IDs and content hashes."""
    root = Path(path)
    paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    documents: list[SourceDocument] = []
    for file_path in paths:
        try:
            documents.extend(_read_file(file_path) or [])
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error) as exc:
            logger.warning("Could not ingest %s: %s", file_path, exc)
    return [document for document in documents if document.text]

