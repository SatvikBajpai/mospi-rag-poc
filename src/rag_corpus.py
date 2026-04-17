"""
Hierarchical RAG over the full MoSPI corpus (v2, hybrid retrieval).

Pipeline:

  query
    -> [vector doc search]  union via RRF  [BM25 doc search]    -> top HIER_TOP_DOCS PDFs
    -> [vector chunk search] union via RRF [BM25 chunk search]  (both filtered to those PDFs)
       -> top HIER_CANDIDATE_CHUNKS chunks
    -> cross-encoder rerank  -> top HIER_FINAL_CHUNKS
    -> Ollama / Gemma generation

Why hybrid: pure vector retrieval misses exact strings (named entities,
numbers, codes). Pure BM25 misses paraphrases. RRF fusion gets both.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from functools import lru_cache

import chromadb
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import (
    BM25_CHUNKS_PATH,
    BM25_DOCS_PATH,
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
    RRF_K,
)

SYSTEM_PROMPT = (
    "You answer questions about official Indian statistics using ONLY the provided "
    "context excerpts from MoSPI publications. Follow these rules strictly:\n"
    "1. If the answer is not explicitly present in the context, reply exactly: "
    "\"I could not find this in the provided MoSPI documents.\" Do not guess.\n"
    "2. Quote exact numbers, percentages, dates, and names as written in the context. "
    "Do not paraphrase numeric values.\n"
    "3. When the context is a table, identify the correct row AND column before "
    "answering. If you cannot unambiguously identify both, decline per rule 1.\n"
    "4. Do not combine numbers from different excerpts unless the question asks for "
    "a comparison and both excerpts are clearly about the same measure.\n"
    "5. Keep the answer short - 1-3 sentences. Do not write a sources list (the caller "
    "adds it)."
)


@dataclass
class CorpusHit:
    text: str
    chunk_id: str
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


@lru_cache(maxsize=1)
def _bm25_docs():
    if not BM25_DOCS_PATH.exists():
        return None
    with BM25_DOCS_PATH.open("rb") as f:
        return pickle.load(f)


@lru_cache(maxsize=1)
def _bm25_chunks():
    if not BM25_CHUNKS_PATH.exists():
        return None
    with BM25_CHUNKS_PATH.open("rb") as f:
        return pickle.load(f)


_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(text or "")]


def _rrf_fuse(ranked_lists: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. Each list is ordered best-first by id.
    Returns (id, score) sorted best-first."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, item_id in enumerate(lst):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


# ---------------- Stage A: hybrid doc retrieval ----------------

def _vector_top_docs(query: str, k: int) -> list[str]:
    emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    res = _docs_col().query(query_embeddings=emb, n_results=k)
    return [m["doc_id"] for m in res["metadatas"][0]]


def _bm25_top_docs(query: str, k: int) -> list[str]:
    bm = _bm25_docs()
    if bm is None or bm["bm25"] is None:
        return []
    scores = bm["bm25"].get_scores(_tokenize(query))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [bm["ids"][i] for i in order]


def retrieve_docs(query: str, k: int = HIER_TOP_DOCS) -> list[str]:
    """Hybrid: vector + BM25, fused via RRF. Each side provides 2k candidates."""
    vec = _vector_top_docs(query, 2 * k)
    bm = _bm25_top_docs(query, 2 * k)
    fused = _rrf_fuse([vec, bm])
    return [doc_id for doc_id, _ in fused[:k]]


# ---------------- Stage B: hybrid chunk retrieval ----------------

def _vector_chunks_in_docs(query: str, doc_ids: list[str], k: int) -> list[tuple[str, dict, str, float]]:
    emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    where = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    res = _chunks_col().query(query_embeddings=emb, n_results=k, where=where)
    out = []
    for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
        out.append((cid, meta, doc, 1.0 - float(dist)))
    return out


def _bm25_chunks_in_docs(query: str, doc_ids: list[str], k: int) -> list[str]:
    bm = _bm25_chunks()
    if bm is None or bm["bm25"] is None:
        return []
    allowed = set(doc_ids)
    scores = bm["bm25"].get_scores(_tokenize(query))
    # filter to allowed docs, sort by score
    order = sorted(
        [i for i in range(len(scores)) if bm["doc_ids"][i] in allowed],
        key=lambda i: scores[i],
        reverse=True,
    )[:k]
    return [bm["ids"][i] for i in order]


def retrieve_chunks(
    query: str,
    doc_ids: list[str],
    k: int = HIER_CANDIDATE_CHUNKS,
) -> list[CorpusHit]:
    """Hybrid chunk retrieval within the candidate doc set."""
    vec_hits = _vector_chunks_in_docs(query, doc_ids, 2 * k)
    vec_ranked_ids = [c[0] for c in vec_hits]
    bm_ranked_ids = _bm25_chunks_in_docs(query, doc_ids, 2 * k)
    fused = _rrf_fuse([vec_ranked_ids, bm_ranked_ids])

    # We need metadata/text for any chunk_id in fused. Vector hits already carry it;
    # for BM25-only hits, fetch from Chroma.
    vec_by_id: dict[str, tuple[dict, str, float]] = {cid: (m, d, s) for cid, m, d, s in vec_hits}
    missing = [cid for cid, _ in fused[:k] if cid not in vec_by_id]
    if missing:
        res = _chunks_col().get(ids=missing, include=["documents", "metadatas"])
        for cid, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            vec_by_id[cid] = (meta, doc, 0.0)

    out: list[CorpusHit] = []
    for cid, score in fused[:k]:
        if cid not in vec_by_id:
            continue
        meta, doc, vec_score = vec_by_id[cid]
        out.append(
            CorpusHit(
                text=doc,
                chunk_id=cid,
                doc_id=meta.get("doc_id", ""),
                parent_title=meta.get("parent_title", ""),
                chapter_title=meta.get("chapter_title", ""),
                filename=meta.get("filename", "?"),
                page=int(meta.get("page", 0)),
                url=meta.get("url", ""),
                score=vec_score,
            )
        )
    return out


# ---------------- Stage C: rerank ----------------

def rerank(query: str, hits: list[CorpusHit], k: int = HIER_FINAL_CHUNKS) -> list[CorpusHit]:
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


# ---------------- Stage D: generation ----------------

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
