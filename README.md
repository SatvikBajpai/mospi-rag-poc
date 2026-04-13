# MoSPI RAG PoC (Small Language Model)

Proof-of-concept RAG over MoSPI press releases (CPI, IIP, PLFS) using a Small Language Model. Goal: show that an SLM + good retrieval can answer questions comparably to the H200-hosted chatbots at a fraction of the cost.

## Stack

- Embeddings: `BAAI/bge-small-en-v1.5` (384-dim, CPU-friendly)
- Vector store: ChromaDB (local persistent)
- Generator SLM: `Qwen/Qwen2.5-1.5B-Instruct` (runs on Mac MPS / CPU)
- Orchestration: plain Python, Typer CLI

Everything runs locally, no external APIs.

## Setup

```bash
cd mospi-rag-poc
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# One-time: extract + embed + index all PDFs in data/raw/
python -m src.cli ingest

# Ask questions
python -m src.cli ask "What was India's CPI inflation in February 2026?"
python -m src.cli ask "Which sector drove IIP growth in December 2025?"
python -m src.cli ask "What is the latest PLFS unemployment rate for urban areas?"

# Drop into an interactive loop
python -m src.cli chat
```

## Corpus

10 PDFs in `data/raw/`:
- CPI press releases: Nov 2025, Dec 2025, Jan 2026, Feb 2026
- IIP press releases: Nov 2025, Dec 2025, Feb 2026
- PLFS quarterly bulletins / press notes: Jul-Sep 2024, Oct-Dec 2024, Oct-Dec 2025

## Layout

```
mospi-rag-poc/
  data/
    raw/           PDFs downloaded from mospi.gov.in
    processed/     extracted text (cached)
  db/              ChromaDB persistent store
  src/
    ingest.py      PDF -> chunks -> embeddings -> Chroma
    rag.py         retrieval + prompt + SLM generation
    cli.py         typer entrypoint
```
