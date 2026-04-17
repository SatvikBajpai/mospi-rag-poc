"""
Manifest-driven corpus ingestion for the full MoSPI scrape (v2).

v2 changes over v1:
  - Doc-level embedding is the mean of a doc's chunk embeddings (whole-doc
    signal) instead of just "title + first 500 words".
  - Each chunk is prefixed with its parent/chapter title at index time so it
    carries its own context when retrieved solo.
  - A BM25 index is built alongside the vector index (at both doc and chunk
    levels) and written to `db/bm25_*.pkl`. This enables hybrid retrieval in
    rag_corpus.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chromadb
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import (
    BM25_CHUNKS_PATH,
    BM25_DOCS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_CHUNKS,
    COLLECTION_DOCS,
    CORPUS_MANIFEST,
    DATA_RAW,
    DB_DIR,
    EMBED_MODEL,
)
from .ingest import chunk_text, extract_pdf_text


@dataclass
class ManifestRow:
    source: str
    parent_id: str
    parent_title: str
    chapter_id: str | None
    chapter_title: str | None
    file_slot: str
    filename: str
    filemime: str
    filesize: int | None
    url: str
    published_date: str | None


def _read_manifest(path: Path) -> Iterator[ManifestRow]:
    if not path.exists():
        raise FileNotFoundError(
            f"manifest not found at {path}. Run `python -m src.scrape_mospi enumerate` first."
        )
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield ManifestRow(**json.loads(line))


def _local_path(row: ManifestRow, raw_root: Path) -> Path:
    parent = row.parent_id or "unknown"
    chapter = row.chapter_id or "0"
    base = Path(row.filename).name.replace("/", "_")
    return raw_root / row.source / parent / f"{chapter}__{base}"


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _doc_id(row: ManifestRow) -> str:
    suffix = hashlib.md5(row.url.encode("utf-8")).hexdigest()[:8]
    return f"{row.source}::{row.parent_id}::{row.chapter_id or '0'}::{row.file_slot}::{suffix}"


def _header(row: ManifestRow) -> str:
    """Context header prepended to each chunk so it carries its source."""
    title = _strip_html(row.parent_title)
    chap = _strip_html(row.chapter_title or "")
    parts = [f"[{title}"]
    if chap:
        parts.append(f" / {chap}")
    parts.append("]")
    return "".join(parts)


_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(text or "")]


def build_corpus_index(
    manifest_path: Path = CORPUS_MANIFEST,
    raw_root: Path = DATA_RAW,
    reset: bool = True,
    batch_size: int = 64,
) -> None:
    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ingest_corpus] loading embedder: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        for c in (COLLECTION_DOCS, COLLECTION_CHUNKS):
            try:
                client.delete_collection(c)
            except Exception:
                pass
        for p in (BM25_DOCS_PATH, BM25_CHUNKS_PATH):
            if p.exists():
                p.unlink()

    docs_col = client.get_or_create_collection(
        name=COLLECTION_DOCS, metadata={"hnsw:space": "cosine"}
    )
    chunks_col = client.get_or_create_collection(
        name=COLLECTION_CHUNKS, metadata={"hnsw:space": "cosine"}
    )

    # BM25 accumulators
    doc_tokens: list[list[str]] = []
    doc_ids_for_bm25: list[str] = []
    chunk_tokens: list[list[str]] = []
    chunk_ids_for_bm25: list[str] = []
    chunk_doc_ids_for_bm25: list[str] = []

    n_processed = 0
    n_skipped = 0
    n_chunks = 0

    # Per-doc batching: accumulate chunks, embed them, then compute doc embedding
    # as the mean of its own chunk embeddings.
    for row in _read_manifest(manifest_path):
        if row.filemime != "application/pdf":
            continue
        local = _local_path(row, raw_root)
        if not local.exists() or local.stat().st_size == 0:
            n_skipped += 1
            continue

        try:
            pages = extract_pdf_text(local)
        except Exception as e:
            print(f"[ingest_corpus] skip {local.name}: extract failed: {e}")
            n_skipped += 1
            continue
        if not pages:
            n_skipped += 1
            continue

        did = _doc_id(row)
        header = _header(row)
        title = _strip_html(row.parent_title)
        chap = _strip_html(row.chapter_title or "")
        meta_base = {
            "doc_id": did,
            "source": row.source,
            "parent_id": row.parent_id,
            "parent_title": title[:300],
            "chapter_id": row.chapter_id or "",
            "chapter_title": chap[:300],
            "filename": row.filename,
            "published_date": row.published_date or "",
            "url": row.url,
        }

        # Chunks for this doc
        local_chunk_ids: list[str] = []
        local_chunk_texts: list[str] = []  # prefixed with header for embedding + retrieval
        local_chunk_raw: list[str] = []     # raw text (used for BM25 tokens)
        local_chunk_metas: list[dict] = []
        for page_no, page_text in pages:
            for i, chunk in enumerate(chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)):
                cid = f"{did}::p{page_no}::c{i}"
                prefixed = f"{header}\n{chunk}"
                local_chunk_ids.append(cid)
                local_chunk_texts.append(prefixed)
                local_chunk_raw.append(chunk)
                local_chunk_metas.append({**meta_base, "page": page_no})

        if not local_chunk_ids:
            n_skipped += 1
            continue

        chunk_embs = embedder.encode(
            local_chunk_texts, batch_size=batch_size, normalize_embeddings=True
        )
        chunks_col.add(
            ids=local_chunk_ids,
            documents=local_chunk_texts,
            metadatas=local_chunk_metas,
            embeddings=chunk_embs.tolist(),
        )
        n_chunks += len(local_chunk_ids)

        # Doc-level embedding = mean of chunk embeddings (renormalized for cosine).
        doc_emb = chunk_embs.mean(axis=0)
        doc_emb = doc_emb / (np.linalg.norm(doc_emb) + 1e-9)
        doc_summary = f"{title}\n{chap}\n" + " ".join(local_chunk_raw)[:2000]
        docs_col.add(
            ids=[did],
            documents=[doc_summary],
            metadatas=[meta_base],
            embeddings=[doc_emb.tolist()],
        )

        # BM25 accumulation
        doc_text_for_bm25 = f"{title} {chap} " + " ".join(local_chunk_raw)
        doc_tokens.append(_tokenize(doc_text_for_bm25))
        doc_ids_for_bm25.append(did)
        for cid, raw, meta in zip(local_chunk_ids, local_chunk_raw, local_chunk_metas):
            chunk_tokens.append(_tokenize(f"{title} {chap} {raw}"))
            chunk_ids_for_bm25.append(cid)
            chunk_doc_ids_for_bm25.append(did)

        n_processed += 1
        if n_processed % 25 == 0:
            print(f"[ingest_corpus] processed={n_processed} chunks={n_chunks} skipped={n_skipped}")

    # Build and persist BM25 indexes
    print(f"[ingest_corpus] building BM25 over {len(doc_tokens)} docs / {len(chunk_tokens)} chunks")
    bm25_docs = {
        "bm25": BM25Okapi(doc_tokens) if doc_tokens else None,
        "ids": doc_ids_for_bm25,
    }
    bm25_chunks = {
        "bm25": BM25Okapi(chunk_tokens) if chunk_tokens else None,
        "ids": chunk_ids_for_bm25,
        "doc_ids": chunk_doc_ids_for_bm25,
    }
    with BM25_DOCS_PATH.open("wb") as f:
        pickle.dump(bm25_docs, f)
    with BM25_CHUNKS_PATH.open("wb") as f:
        pickle.dump(bm25_chunks, f)

    print(
        f"[ingest_corpus] done. docs={docs_col.count()} chunks={chunks_col.count()} "
        f"skipped={n_skipped}"
    )


if __name__ == "__main__":
    build_corpus_index()
