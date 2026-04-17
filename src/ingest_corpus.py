"""
Manifest-driven corpus ingestion for the full MoSPI scrape.

Reads `data/raw/manifest.jsonl` (produced by `scrape_mospi`) and indexes every
local PDF into two ChromaDB collections:

  - mospi_corpus_docs    : one row per PDF, embedding = "title + chapter + first ~500 words"
  - mospi_corpus_chunks  : page-level chunks, with doc_id FK so chunks can be
                           filtered to a candidate doc set during retrieval

This is the doc-level half of the hierarchical retrieval discussed in the design:
chunk-search across 30k+ chunks is noisy, but doc-level search narrows to ~5
candidate PDFs first, then chunk search runs only within those.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_CHUNKS,
    COLLECTION_DOCS,
    CORPUS_MANIFEST,
    DATA_RAW,
    DB_DIR,
    DOC_SUMMARY_WORDS,
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
    """Mirror the layout that scrape_mospi.download writes."""
    parent = row.parent_id or "unknown"
    chapter = row.chapter_id or "0"
    base = Path(row.filename).name.replace("/", "_")
    return raw_root / row.source / parent / f"{chapter}__{base}"


def _strip_html(s: str) -> str:
    """Manifest titles sometimes contain HTML tags. Cheap stripper, no bs4 dep."""
    import re
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _doc_id(row: ManifestRow) -> str:
    # URL-hash suffix guarantees uniqueness even when (parent_id, chapter_id, file_slot)
    # collides — happens with sub_chapters that share an empty chapter_id.
    suffix = hashlib.md5(row.url.encode("utf-8")).hexdigest()[:8]
    return f"{row.source}::{row.parent_id}::{row.chapter_id or '0'}::{row.file_slot}::{suffix}"


def _doc_summary_text(row: ManifestRow, full_text: str) -> str:
    """Front-loaded text used to compute the doc-level embedding.
    Title + chapter + first DOC_SUMMARY_WORDS words is a strong signal in practice."""
    title = _strip_html(row.parent_title)
    chapter = _strip_html(row.chapter_title or "")
    head = " ".join(full_text.split()[:DOC_SUMMARY_WORDS])
    parts = [f"Title: {title}"]
    if chapter:
        parts.append(f"Chapter: {chapter}")
    parts.append(head)
    return "\n\n".join(parts)


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
    docs_col = client.get_or_create_collection(
        name=COLLECTION_DOCS, metadata={"hnsw:space": "cosine"}
    )
    chunks_col = client.get_or_create_collection(
        name=COLLECTION_CHUNKS, metadata={"hnsw:space": "cosine"}
    )

    n_processed = 0
    n_skipped = 0
    n_chunks = 0

    doc_buf_ids: list[str] = []
    doc_buf_texts: list[str] = []
    doc_buf_metas: list[dict] = []

    chunk_buf_ids: list[str] = []
    chunk_buf_texts: list[str] = []
    chunk_buf_metas: list[dict] = []

    def _flush_docs():
        if not doc_buf_ids:
            return
        embs = embedder.encode(doc_buf_texts, batch_size=batch_size, normalize_embeddings=True).tolist()
        docs_col.add(ids=doc_buf_ids, documents=doc_buf_texts, metadatas=doc_buf_metas, embeddings=embs)
        doc_buf_ids.clear(); doc_buf_texts.clear(); doc_buf_metas.clear()

    def _flush_chunks():
        if not chunk_buf_ids:
            return
        embs = embedder.encode(chunk_buf_texts, batch_size=batch_size, normalize_embeddings=True).tolist()
        chunks_col.add(ids=chunk_buf_ids, documents=chunk_buf_texts, metadatas=chunk_buf_metas, embeddings=embs)
        chunk_buf_ids.clear(); chunk_buf_texts.clear(); chunk_buf_metas.clear()

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

        full_text = "\n\n".join(t for _, t in pages)
        doc_id = _doc_id(row)
        meta_base = {
            "doc_id": doc_id,
            "source": row.source,
            "parent_id": row.parent_id,
            "parent_title": _strip_html(row.parent_title)[:300],
            "chapter_id": row.chapter_id or "",
            "chapter_title": _strip_html(row.chapter_title or "")[:300],
            "filename": row.filename,
            "published_date": row.published_date or "",
            "url": row.url,
        }

        doc_buf_ids.append(doc_id)
        doc_buf_texts.append(_doc_summary_text(row, full_text))
        doc_buf_metas.append(meta_base)

        for page_no, page_text in pages:
            for i, chunk in enumerate(chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)):
                cid = f"{doc_id}::p{page_no}::c{i}"
                chunk_buf_ids.append(cid)
                chunk_buf_texts.append(chunk)
                chunk_buf_metas.append({**meta_base, "page": page_no})
                n_chunks += 1

        n_processed += 1
        if n_processed % 25 == 0:
            print(f"[ingest_corpus] processed={n_processed} chunks={n_chunks} skipped={n_skipped}")
            _flush_docs(); _flush_chunks()

    _flush_docs(); _flush_chunks()
    print(
        f"[ingest_corpus] done. docs={docs_col.count()} chunks={chunks_col.count()} "
        f"skipped={n_skipped}"
    )


if __name__ == "__main__":
    build_corpus_index()
