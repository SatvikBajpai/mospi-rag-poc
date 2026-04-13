"""Typer CLI: ingest + ask + chat."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .ingest import build_index
from .rag import ask as rag_ask

app = typer.Typer(add_completion=False, help="MoSPI RAG PoC (SLM-powered)")
console = Console()


@app.command()
def ingest(reset: bool = typer.Option(True, help="Drop and rebuild the collection")):
    """Extract, chunk, embed, and index all PDFs in data/raw/."""
    build_index(reset=reset)


def _show(answer: str, hits) -> None:
    console.print(Panel(answer, title="Answer", border_style="green"))
    table = Table(title="Retrieved chunks", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("source")
    table.add_column("page", justify="right")
    table.add_column("score", justify="right")
    for i, h in enumerate(hits, 1):
        table.add_row(str(i), h.source, str(h.page), f"{h.score:.3f}")
    console.print(table)


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural language question"),
    topic: str = typer.Option(None, help="Filter by topic: CPI, IIP, PLFS"),
    k: int = typer.Option(5, help="Top-k chunks to retrieve"),
):
    """Ask a single question."""
    answer, hits = rag_ask(question, topic=topic, k=k)
    _show(answer, hits)


@app.command()
def chat(topic: str = typer.Option(None, help="Filter by topic: CPI, IIP, PLFS")):
    """Interactive loop."""
    console.print("[bold cyan]MoSPI RAG chat. Ctrl-C to exit.[/bold cyan]")
    while True:
        try:
            q = console.input("[bold]you> [/bold]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nbye.")
            return
        if not q:
            continue
        answer, hits = rag_ask(q, topic=topic)
        _show(answer, hits)


if __name__ == "__main__":
    app()
