"""Retrieval + SLM generation via Ollama."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
import requests
from sentence_transformers import SentenceTransformer

from .config import COLLECTION, DB_DIR, EMBED_MODEL, MAX_NEW_TOKENS, OLLAMA_MODEL, OLLAMA_URL, TOP_K

SYSTEM_PROMPT = (
    "You answer questions about Indian official statistics (CPI, IIP, PLFS) "
    "using ONLY the provided context from MoSPI press releases.\n"
    "- If the answer is not in the context, reply: "
    "\"I could not find this in the provided MoSPI documents.\"\n"
    "- Be concise. Quote exact numbers, sector names, and dates from the context.\n"
    "- Do not invent figures. Do not write a sources list - the caller adds it."
)


@dataclass
class Retrieved:
    text: str
    source: str
    page: int
    score: float


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client.get_collection(COLLECTION)


def retrieve(query: str, k: int = TOP_K, topic: str | None = None) -> list[Retrieved]:
    emb = _embedder().encode([query], normalize_embeddings=True).tolist()
    where = {"topic": topic} if topic else None
    res = _collection().query(
        query_embeddings=emb,
        n_results=k,
        where=where,
    )
    out: list[Retrieved] = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(
            Retrieved(
                text=doc,
                source=meta.get("source", "?"),
                page=int(meta.get("page", 0)),
                score=1.0 - float(dist),
            )
        )
    return out


def build_prompt(query: str, hits: list[Retrieved]) -> list[dict]:
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        context_blocks.append(
            f"[{i}] source={h.source} page={h.page}\n{h.text}"
        )
    context = "\n\n".join(context_blocks)
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def generate(messages: list[dict]) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {
                "num_predict": MAX_NEW_TOKENS,
                "temperature": 0.0,
            },
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def _format_sources(hits: list[Retrieved]) -> str:
    seen: dict[tuple[str, int], None] = {}
    for h in hits:
        seen.setdefault((h.source, h.page), None)
    lines = [f"  - {src} (page {page})" for (src, page) in seen]
    return "Sources:\n" + "\n".join(lines)


def ask(query: str, topic: str | None = None, k: int = TOP_K) -> tuple[str, list[Retrieved]]:
    hits = retrieve(query, k=k, topic=topic)
    messages = build_prompt(query, hits)
    answer = generate(messages).strip()
    answer = f"{answer}\n\n{_format_sources(hits)}"
    return answer, hits
