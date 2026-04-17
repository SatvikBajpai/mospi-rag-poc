"""Typer CLI: ingest + ask + chat (PoC: 10-PDF flat) and ingest-corpus + ask-corpus (full corpus: hierarchical)."""
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .eval_corpus import inspect_index, run_eval
from .ingest import build_index
from .ingest_corpus import build_corpus_index
from .rag import ask as rag_ask
from .rag_corpus import ask as corpus_ask

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


# ---------- corpus-scale commands (manifest-driven, hierarchical) ----------

def _show_corpus(answer: str, hits) -> None:
    console.print(Panel(answer, title="Answer", border_style="green"))
    table = Table(title="Retrieved chunks (post-rerank)", show_lines=False)
    table.add_column("#", justify="right")
    table.add_column("doc")
    table.add_column("page", justify="right")
    table.add_column("vec", justify="right")
    table.add_column("rerank", justify="right")
    for i, h in enumerate(hits, 1):
        label = h.parent_title[:60] + ("..." if len(h.parent_title) > 60 else "")
        if h.chapter_title:
            label = f"{label} / {h.chapter_title[:30]}"
        rer = f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-"
        table.add_row(str(i), label, str(h.page), f"{h.score:.3f}", rer)
    console.print(table)


@app.command(name="ingest-corpus")
def ingest_corpus(reset: bool = typer.Option(True, help="Drop and rebuild the corpus collections.")):
    """Build doc + chunk indices from data/raw/manifest.jsonl."""
    build_corpus_index(reset=reset)


@app.command(name="ask-corpus")
def ask_corpus(
    question: str = typer.Argument(..., help="Natural language question"),
    top_docs: int = typer.Option(5, help="Stage 1: candidate PDFs from doc-level search."),
    candidate_chunks: int = typer.Option(20, help="Stage 2: candidate chunks before reranker."),
    final_chunks: int = typer.Option(4, help="Stage 3: chunks fed to the LLM."),
    no_rerank: bool = typer.Option(False, "--no-rerank", help="Skip the cross-encoder rerank stage."),
):
    """Ask a single question against the full corpus (hierarchical retrieval)."""
    answer, hits = corpus_ask(
        question,
        top_docs=top_docs,
        candidate_chunks=candidate_chunks,
        final_chunks=final_chunks,
        use_rerank=not no_rerank,
    )
    _show_corpus(answer, hits)


@app.command(name="chat-corpus")
def chat_corpus(no_rerank: bool = typer.Option(False, "--no-rerank")):
    """Interactive loop against the full corpus."""
    console.print("[bold cyan]MoSPI corpus chat (hierarchical). Ctrl-C to exit.[/bold cyan]")
    while True:
        try:
            q = console.input("[bold]you> [/bold]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nbye.")
            return
        if not q:
            continue
        answer, hits = corpus_ask(q, use_rerank=not no_rerank)
        _show_corpus(answer, hits)


@app.command(name="inspect-corpus")
def inspect_corpus():
    """Show what's currently in the corpus index (counts, sources, sample titles)."""
    inspect_index()


@app.command(name="eval-corpus")
def eval_corpus(
    n: int = typer.Option(20, help="Number of synthetic questions to evaluate."),
    top_docs: int = typer.Option(5),
    candidate_chunks: int = typer.Option(20),
    final_chunks: int = typer.Option(4),
    no_rerank: bool = typer.Option(False, "--no-rerank"),
    no_answer: bool = typer.Option(False, "--no-answer", help="Skip generating answers (faster)."),
    seed: int = typer.Option(42),
):
    """Synthetic eval: Gemma generates questions from random chunks, then we
    measure whether retrieval finds those chunks back. Writes eval/corpus_eval_report.md."""
    run_eval(
        n=n,
        top_docs=top_docs,
        candidate_chunks=candidate_chunks,
        final_chunks=final_chunks,
        use_rerank=not no_rerank,
        generate_answer=not no_answer,
        seed=seed,
    )


if __name__ == "__main__":
    app()
