"""
Hierarchical RAG over the full MoSPI corpus.

Two-stage retrieval to keep accuracy from collapsing as the corpus scales:

  query
    -> doc-level vector search   (top HIER_TOP_DOCS PDFs)
    -> chunk-level vector search  (top HIER_CANDIDATE_CHUNKS chunks, filtered to those PDFs)
    -> cross-encoder rerank       (top HIER_FINAL_CHUNKS)
    -> Ollama generation          (Gemma 3 4B by default)

Dropping the reranker (`--no-rerank`) skips stage 3 and feeds the top
HIER_FINAL_CHUNKS chunks straight from the bi-encoder. The reranker adds
~50-200ms on CPU but materially improves precision when the doc is long.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import (
    COLLECTION_CHUNKS,
    COLLECTION_DOCS,
    DB_DIR,
    EMBED_MODEL,
    HIER_CANDIDATE_CHUNKS,
    HIER_FINAL_CHUNKS,
    HIER_TOP_DOCS,
    MAX_NEW_TOKENS,
    OLLAMA_MODEL,
    OLLAMA_URL,
    RERANKER_MODEL,
)

SYSTEM_PROMPT = (
    "You answer questions about official Indian statistics using ONLY the provided "
    "context excerpts from MoSPI publications.\n"
    "- If the answer is not in the context, reply: "
    "\"I could not find this in the provided MoSPI documents.\"\n"
    "- Be concise. Quote exact numbers, dates, and series names from the context.\n"
    "- Do not invent figures. Do not write a sources list - the caller adds it."
)


@dataclass
class CorpusHit:
    text: str
    doc_id: str
    parent_title: str
    chapter_title: str
    filename: str
    page: int
    url: str
    score: float
    rerank_score: float | None = None


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(DB_DIR))


def _docs_col():
    return _client().get_collection(COLLECTION_DOCS)


def _chunks_col():
    return _client().get_collection(COLLECTION_CHUNKS)


def retrieve_docs(query: str, k: int = HIER_TOP_DOCS) -> list[str]:
    """Stage 1: doc-level search. Returns top-k doc_ids."""
    emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    res = _docs_col().query(query_embeddings=emb, n_results=k)
    metas = res["metadatas"][0]
    return [m["doc_id"] for m in metas]


def retrieve_chunks(
    query: str,
    doc_ids: list[str],
    k: int = HIER_CANDIDATE_CHUNKS,
) -> list[CorpusHit]:
    """Stage 2: chunk-level search restricted to the candidate doc_ids."""
    emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    res = _chunks_col().query(query_embeddings=emb, n_results=k, where=where)
    out: list[CorpusHit] = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append(
            CorpusHit(
                text=doc,
                doc_id=meta.get("doc_id", ""),
                parent_title=meta.get("parent_title", ""),
                chapter_title=meta.get("chapter_title", ""),
                filename=meta.get("filename", "?"),
                page=int(meta.get("page", 0)),
                url=meta.get("url", ""),
                score=1.0 - float(dist),
            )
        )
    return out


def rerank(query: str, hits: list[CorpusHit], k: int = HIER_FINAL_CHUNKS) -> list[CorpusHit]:
    """Stage 3: cross-encoder reranking."""
    if not hits:
        return hits
    pairs = [(query, h.text) for h in hits]
    scores = _reranker().predict(pairs).tolist()
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    hits.sort(key=lambda h: (h.rerank_score if h.rerank_score is not None else h.score), reverse=True)
    return hits[:k]


def hierarchical_retrieve(
    query: str,
    top_docs: int = HIER_TOP_DOCS,
    candidate_chunks: int = HIER_CANDIDATE_CHUNKS,
    final_chunks: int = HIER_FINAL_CHUNKS,
    use_rerank: bool = True,
) -> list[CorpusHit]:
    doc_ids = retrieve_docs(query, k=top_docs)
    candidates = retrieve_chunks(query, doc_ids, k=candidate_chunks)
    if use_rerank:
        return rerank(query, candidates, k=final_chunks)
    return candidates[:final_chunks]


def _build_prompt(query: str, hits: list[CorpusHit]) -> list[dict]:
    blocks = []
    for i, h in enumerate(hits, start=1):
        header = f"[{i}] {h.parent_title}"
        if h.chapter_title:
            header += f" / {h.chapter_title}"
        header += f" (page {h.page})"
        blocks.append(f"{header}\n{h.text}")
    context = "\n\n".join(blocks)
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _generate(messages: list[dict]) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"num_predict": MAX_NEW_TOKENS, "temperature": 0.0},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _format_sources(hits: list[CorpusHit]) -> str:
    seen: dict[tuple[str, int], str] = {}
    for h in hits:
        key = (h.filename, h.page)
        if key not in seen:
            label = h.parent_title
            if h.chapter_title:
                label = f"{label} / {h.chapter_title}"
            seen[key] = label
    lines = [f"  - {label} - {fn} (page {pg})" for (fn, pg), label in seen.items()]
    return "Sources:\n" + "\n".join(lines)


def ask(
    query: str,
    top_docs: int = HIER_TOP_DOCS,
    candidate_chunks: int = HIER_CANDIDATE_CHUNKS,
    final_chunks: int = HIER_FINAL_CHUNKS,
    use_rerank: bool = True,
) -> tuple[str, list[CorpusHit]]:
    hits = hierarchical_retrieve(
        query,
        top_docs=top_docs,
        candidate_chunks=candidate_chunks,
        final_chunks=final_chunks,
        use_rerank=use_rerank,
    )
    answer = _generate(_build_prompt(query, hits))
    answer = f"{answer}\n\n{_format_sources(hits)}"
    return answer, hits
