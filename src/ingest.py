"""Extract text from PDFs, chunk, embed, and upsert into ChromaDB."""
from __future__ import annotations

import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION,
    DATA_PROCESSED,
    DATA_RAW,
    DB_DIR,
    EMBED_MODEL,
)


def extract_pdf_text(pdf_path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text]
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if chunk.strip():
            chunks.append(chunk)
        if start + size >= len(words):
            break
    return chunks


def infer_topic(filename: str) -> str:
    name = filename.upper()
    if name.startswith("CPI"):
        return "CPI"
    if name.startswith("IIP"):
        return "IIP"
    if name.startswith("PLFS"):
        return "PLFS"
    return "OTHER"


def build_index(reset: bool = True) -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[ingest] loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name=COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    pdfs = sorted(DATA_RAW.glob("*.pdf"))
    if not pdfs:
        print(f"[ingest] no PDFs found in {DATA_RAW}")
        return

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict] = []

    for pdf in pdfs:
        print(f"[ingest] processing {pdf.name}")
        pages = extract_pdf_text(pdf)
        full_text = "\n\n".join(t for _, t in pages)
        (DATA_PROCESSED / f"{pdf.stem}.txt").write_text(full_text, encoding="utf-8")

        topic = infer_topic(pdf.name)
        for page_no, page_text in pages:
            for i, chunk in enumerate(chunk_text(page_text)):
                ids.append(f"{pdf.stem}::p{page_no}::c{i}")
                docs.append(chunk)
                metas.append(
                    {
                        "source": pdf.name,
                        "topic": topic,
                        "page": page_no,
                    }
                )

    print(f"[ingest] embedding {len(docs)} chunks")
    embeddings = embedder.encode(
        docs, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).tolist()

    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    print(f"[ingest] done. collection size = {collection.count()}")


if __name__ == "__main__":
    build_index()
