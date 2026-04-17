"""
Synthetic eval for the corpus RAG.

Without a hand-labeled eval set we can still measure retrieval quality:

  1. Sample N random chunks from the indexed corpus.
  2. Ask Gemma to generate a specific factual question that each chunk answers.
  3. Run that question through the full RAG pipeline.
  4. Check whether the originating chunk and doc come back in the retrieved
     candidates - this gives us doc_recall@k and chunk_recall@k.
  5. Generate the final answer with Gemma and dump everything to a report
     so you can spot-check by reading.

This is a "passage-level" eval - it tells you whether retrieval can find the
right passage when asked about it. It does NOT measure whether Gemma writes a
correct or grounded answer; for that, read the markdown report.

Outputs:
  eval/corpus_eval_results.jsonl  - one JSON line per question
  eval/corpus_eval_report.md      - human-readable side-by-side
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import chromadb
import requests
import typer
from rich.console import Console
from rich.table import Table

from .config import (
    COLLECTION_CHUNKS,
    COLLECTION_DOCS,
    DB_DIR,
    HIER_CANDIDATE_CHUNKS,
    HIER_FINAL_CHUNKS,
    HIER_TOP_DOCS,
    MAX_NEW_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_URL,
)
from .rag_corpus import _generate, hierarchical_retrieve

console = Console()

QUESTION_GEN_PROMPT = (
    "You are creating a question for a search-evaluation test.\n"
    "Read the passage below, then write ONE specific, factual question that the passage "
    "directly answers. The question must be self-contained (do not refer to 'the passage' "
    "or 'the document'). Use proper nouns, numbers, and dates from the passage so the "
    "question is unambiguous.\n\n"
    "Output ONLY the question on a single line, nothing else."
)


@dataclass
class EvalCase:
    question: str
    expected_chunk_id: str
    expected_doc_id: str
    expected_parent_title: str
    expected_filename: str
    expected_page: int
    expected_chunk_text: str
    retrieved_chunk_ids: list[str]
    retrieved_doc_ids: list[str]
    chunk_recall: bool
    doc_recall: bool
    answer: str | None = None


def _client():
    return chromadb.PersistentClient(path=str(DB_DIR))


def _generate_question(chunk_text: str) -> str:
    msgs = [
        {"role": "system", "content": QUESTION_GEN_PROMPT},
        {"role": "user", "content": chunk_text[:3000]},
    ]
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": OLLAMA_MODEL, "messages": msgs, "stream": False,
              "options": {"num_predict": 120, "temperature": 0.0}},
        timeout=600,
    )
    resp.raise_for_status()
    q = resp.json()["message"]["content"].strip()
    # Strip common preamble fluff
    q = re.sub(r'^(Question:|Q:)\s*', '', q, flags=re.IGNORECASE).strip()
    q = q.split("\n")[0].strip()
    return q


def _sample_chunks(n: int, min_words: int = 80, seed: int = 42) -> list[dict]:
    """Pick N random chunks of decent length (skip short or table-only chunks)."""
    rng = random.Random(seed)
    col = _client().get_collection(COLLECTION_CHUNKS)
    total = col.count()
    if total == 0:
        raise RuntimeError("chunks collection is empty - run ingest-corpus first")
    # Pull a generous oversample so we can filter
    want_pool = min(total, max(n * 8, 200))
    offsets = sorted(rng.sample(range(total), want_pool))
    res = col.get(limit=want_pool, include=["documents", "metadatas"])
    pool = []
    for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
        if len(doc.split()) >= min_words:
            pool.append({"chunk_id": cid, "text": doc, "meta": meta})
    rng.shuffle(pool)
    return pool[:n]


def _run_one(sample: dict, top_docs: int, candidate_chunks: int, final_chunks: int,
             use_rerank: bool, generate_answer: bool) -> EvalCase:
    text = sample["text"]
    meta = sample["meta"]
    question = _generate_question(text)
    hits = hierarchical_retrieve(
        question, top_docs=top_docs, candidate_chunks=candidate_chunks,
        final_chunks=final_chunks, use_rerank=use_rerank,
    )
    # Note: hierarchical_retrieve only returns the FINAL (post-rerank) hits.
    # For a proper recall measurement we want the candidate set too. So we
    # also peek at the candidate doc set via a fresh retrieve_docs call.
    from .rag_corpus import retrieve_chunks, retrieve_docs
    cand_doc_ids = retrieve_docs(question, k=top_docs)
    cand_chunks = retrieve_chunks(question, cand_doc_ids, k=candidate_chunks)
    cand_chunk_ids = [c.text for c in cand_chunks]  # using text since CorpusHit has no chunk_id field
    # Compare on chunk text equality as a proxy for chunk_id (same content == same chunk)
    chunk_hit = any(text == c.text for c in cand_chunks)
    doc_hit = meta["doc_id"] in cand_doc_ids

    answer = None
    if generate_answer:
        from .rag_corpus import _build_prompt
        msgs = _build_prompt(question, hits)
        try:
            answer = _generate(msgs)
        except Exception as e:
            answer = f"<generation failed: {e}>"

    return EvalCase(
        question=question,
        expected_chunk_id=sample["chunk_id"],
        expected_doc_id=meta["doc_id"],
        expected_parent_title=meta.get("parent_title", "?"),
        expected_filename=meta.get("filename", "?"),
        expected_page=int(meta.get("page", 0)),
        expected_chunk_text=text[:600],
        retrieved_chunk_ids=[],
        retrieved_doc_ids=cand_doc_ids,
        chunk_recall=chunk_hit,
        doc_recall=doc_hit,
        answer=answer,
    )


def _write_report(cases: list[EvalCase], out_md: Path) -> None:
    lines = ["# Corpus RAG eval report", ""]
    n = len(cases)
    doc_recall = sum(1 for c in cases if c.doc_recall) / n
    chunk_recall = sum(1 for c in cases if c.chunk_recall) / n
    lines += [
        f"- Cases: **{n}**",
        f"- doc_recall@top-docs: **{doc_recall:.1%}**  (right PDF reached the candidate set)",
        f"- chunk_recall@candidates: **{chunk_recall:.1%}**  (originating chunk reached the candidate set)",
        "",
    ]
    for i, c in enumerate(cases, 1):
        lines += [
            f"## Q{i}  doc_recall={'YES' if c.doc_recall else 'NO'}  chunk_recall={'YES' if c.chunk_recall else 'NO'}",
            "",
            f"**Question:** {c.question}",
            "",
            f"**Expected source:** {c.expected_parent_title} - {c.expected_filename} (page {c.expected_page})",
            "",
            f"**Expected passage (truncated):**",
            f"> {c.expected_chunk_text.replace(chr(10), ' ')}",
            "",
            f"**Retrieved candidate doc_ids:** `{c.retrieved_doc_ids}`",
            "",
        ]
        if c.answer:
            lines += [f"**Generated answer:**", "", f"> {c.answer.replace(chr(10), chr(10) + '> ')}", ""]
        lines.append("---\n")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def run_eval(
    n: int = 20,
    top_docs: int = HIER_TOP_DOCS,
    candidate_chunks: int = HIER_CANDIDATE_CHUNKS,
    final_chunks: int = HIER_FINAL_CHUNKS,
    use_rerank: bool = True,
    generate_answer: bool = True,
    seed: int = 42,
    out_dir: Path = Path("eval"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = _sample_chunks(n, seed=seed)
    if len(samples) < n:
        console.print(f"[yellow]warning: only {len(samples)} usable chunks found (wanted {n})[/yellow]")

    cases: list[EvalCase] = []
    for i, s in enumerate(samples, 1):
        console.print(f"[dim]case {i}/{len(samples)} - generating question...[/dim]")
        t0 = time.time()
        try:
            case = _run_one(s, top_docs, candidate_chunks, final_chunks, use_rerank, generate_answer)
        except Exception as e:
            console.print(f"[red]case {i} failed: {e}[/red]")
            continue
        cases.append(case)
        dt = time.time() - t0
        console.print(
            f"  Q: {case.question[:90]}{'...' if len(case.question)>90 else ''}\n"
            f"  doc_recall={'[green]YES[/green]' if case.doc_recall else '[red]NO[/red]'}  "
            f"chunk_recall={'[green]YES[/green]' if case.chunk_recall else '[red]NO[/red]'}  "
            f"({dt:.1f}s)"
        )

    out_jsonl = out_dir / "corpus_eval_results.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    out_md = out_dir / "corpus_eval_report.md"
    _write_report(cases, out_md)

    n_ok = len(cases)
    if n_ok == 0:
        console.print("[red]no successful cases - is Ollama running?[/red]")
        return
    doc_recall = sum(1 for c in cases if c.doc_recall) / n_ok
    chunk_recall = sum(1 for c in cases if c.chunk_recall) / n_ok
    table = Table(title="Eval summary", show_lines=False)
    table.add_column("metric"); table.add_column("value", justify="right")
    table.add_row("cases", str(n_ok))
    table.add_row("doc_recall", f"{doc_recall:.1%}")
    table.add_row("chunk_recall", f"{chunk_recall:.1%}")
    console.print(table)
    console.print(f"[green]wrote {out_jsonl} and {out_md}[/green]")


def inspect_index() -> None:
    """Show what's actually indexed - useful before running eval."""
    client = _client()
    docs = client.get_collection(COLLECTION_DOCS)
    chunks = client.get_collection(COLLECTION_CHUNKS)
    n_docs = docs.count(); n_chunks = chunks.count()
    if n_docs == 0:
        console.print("[red]docs collection is empty - run `ingest-corpus` first[/red]")
        return

    res = docs.get(limit=n_docs, include=["metadatas"])
    by_source = Counter(m.get("source", "?") for m in res["metadatas"])

    table = Table(title="Indexed corpus")
    table.add_column("metric"); table.add_column("value", justify="right")
    table.add_row("PDFs (docs)", str(n_docs))
    table.add_row("chunks", str(n_chunks))
    table.add_row("avg chunks/doc", f"{n_chunks / n_docs:.1f}")
    for src, cnt in sorted(by_source.items(), key=lambda kv: -kv[1]):
        table.add_row(f"  source={src}", str(cnt))
    console.print(table)

    console.print("\n[bold]Sample titles:[/bold]")
    for m in res["metadatas"][:10]:
        title = m.get("parent_title", "?")
        chap = m.get("chapter_title", "")
        line = f"  - [{m.get('source','?')}] {title}"
        if chap:
            line += f" / {chap}"
        console.print(line)
