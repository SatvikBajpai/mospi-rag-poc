# MoSPI RAG PoC (Small Language Model)

Proof-of-concept RAG over MoSPI press releases (CPI, IIP, PLFS) using a Small Language Model via Ollama. Goal: show that an SLM + good retrieval can answer questions comparably to the H200-hosted chatbots at a fraction of the cost.

## Stack

- Embeddings: `BAAI/bge-small-en-v1.5` (384-dim, CPU-friendly)
- Vector store: ChromaDB (local persistent)
- Generator SLM: any Ollama model (default: `smollm2:1.7b`)
- Orchestration: plain Python, Typer CLI

Everything runs locally, no external APIs, no GPU required.

## Prerequisites

Install Ollama: https://ollama.com/download

```powershell
# Pull a model (pick one)
ollama pull smollm2:1.7b        # 1.2 GB, default
ollama pull phi4-mini            # 2.4 GB, best quality
ollama pull smollm3              # 1.9 GB, good balance
ollama pull llama3.2:3b          # 2 GB, solid all-rounder
```

## Setup

```powershell
git clone https://github.com/SatvikBajpai/mospi-rag-poc.git
cd mospi-rag-poc
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```powershell
# One-time: extract + embed + index all PDFs in data/raw/
python -m src.cli ingest

# Ask questions
python -m src.cli ask "What was India's CPI inflation in February 2026?"
python -m src.cli ask "Which sector drove IIP growth in December 2025?"
python -m src.cli ask "What is the latest PLFS unemployment rate?"

# Interactive chat
python -m src.cli chat
```

## Swap models

Just set an environment variable - no code changes needed:

```powershell
set OLLAMA_MODEL=phi4-mini
python -m src.cli ask "What was India's CPI inflation in February 2026?"

set OLLAMA_MODEL=smollm3
python -m src.cli ask "What was India's CPI inflation in February 2026?"
```

## Run the eval

16 Q&A pairs with ground-truth answers from the press releases:

```powershell
python -m eval.run_eval
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
    config.py      all settings (env-var overridable)
    ingest.py      PDF -> chunks -> embeddings -> Chroma
    rag.py         retrieval + Ollama SLM generation
    cli.py         typer entrypoint
  eval/
    eval_set.json  16 ground-truth Q&A pairs
    run_eval.py    automated eval runner
```

## Results so far

| Model | Answer accuracy | Retrieval accuracy | Avg latency |
|---|---|---|---|
| Qwen2.5-0.5B (transformers, MPS) | 11/16 (69%) | 16/16 (100%) | ~27s |
| SmolLM2-1.7B (transformers, CPU) | 1/1 tested | 1/1 (100%) | ~2-5 min |
| SmolLM2-1.7B (Ollama, CPU) | TBD | TBD | ~10-20s expected |

Retrieval is the strong link (100%). Generator quality scales with model size.
