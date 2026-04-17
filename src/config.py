import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DB_DIR = ROOT / "db"

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "smollm2:1.7b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
COLLECTION = "mospi_press_releases"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "300"))

# Corpus-scale RAG (manifest-driven, hierarchical retrieval + reranker)
CORPUS_MANIFEST = DATA_RAW / "manifest.jsonl"
COLLECTION_DOCS = "mospi_corpus_docs"
COLLECTION_CHUNKS = "mospi_corpus_chunks"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
HIER_TOP_DOCS = int(os.environ.get("HIER_TOP_DOCS", "5"))
HIER_CANDIDATE_CHUNKS = int(os.environ.get("HIER_CANDIDATE_CHUNKS", "20"))
HIER_FINAL_CHUNKS = int(os.environ.get("HIER_FINAL_CHUNKS", "4"))
DOC_SUMMARY_WORDS = 500  # words from start of doc used to compute doc-level embedding
