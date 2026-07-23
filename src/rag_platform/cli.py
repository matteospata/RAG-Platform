from __future__ import annotations

import json
from pathlib import Path

import typer

from .container import build_container
from .logging_config import configure_logging
from .pipeline import IndexingPipeline

app = typer.Typer(help="CLI for the RAG platform.", no_args_is_help=True)


@app.command()
def ingest(path: str = typer.Option("data/knowledge_base", help="File or directory to index.")) -> None:
    """Run ingestion, chunking, embedding, and idempotent upsert."""
    current = build_container()
    configure_logging(current.settings.log_level)
    source = Path(path)
    if not source.exists():
        raise typer.BadParameter(f"Path not found: {path}")
    run = IndexingPipeline(current.store, current.embeddings, current.settings.chunk_size, current.settings.chunk_overlap).run(source)
    typer.echo(json.dumps(run.model_dump(mode="json"), indent=2, ensure_ascii=False))
    current.store.close()


@app.command()
def ask(question: str = typer.Argument(..., help="Question to ask the knowledge base.")) -> None:
    """Run retrieval and generate an answer with citations."""
    current = build_container()
    result = current.rag.ask(question)
    typer.echo(result.answer)
    typer.echo("\nSources:")
    for index, citation in enumerate(result.citations, 1):
        typer.echo(f"[S{index}] {citation.title} — {citation.source} (score={citation.score:.3f})")
    current.store.close()


if __name__ == "__main__":
    app()
