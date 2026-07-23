.PHONY: install test lint format run ingest ask

install:
	python -m pip install -e '.[dev,llm]'

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

run:
	uvicorn rag_platform.api:app --reload --host 0.0.0.0 --port 8000

ingest:
	python -m rag_platform.cli ingest --path data/knowledge_base

ask:
	python -m rag_platform.cli ask "How is data managed?"
