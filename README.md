# MoSPI RAG PoC (Small Language Model)

Proof-of-concept RAG over MoSPI press releases (CPI, IIP, PLFS) using a Small Language Model. Goal: show that an SLM + good retrieval can answer questions comparably to the H200-hosted chatbots at a fraction of the cost.

## Stack

- Embeddings: `BAAI/bge-small-en-v1.5` (384-dim, CPU-friendly)
- Vector store: ChromaDB (local persistent)
- Generator SLM: configurable via env var (default: `HuggingFaceTB/SmolLM2-1.7B-Instruct`)
- Orchestration: plain Python, Typer CLI

Everything runs locally, no external APIs.

## Setup

```bash
git clone https://github.com/SatvikBajpai/mospi-rag-poc.git
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

## Swap the SLM model

All config is overridable via environment variables:

```bash
# Try different models
export LLM_MODEL="microsoft/Phi-4-mini-instruct"       # 3.8B, best benchmarks
export LLM_MODEL="HuggingFaceTB/SmolLM3-3B"            # 3B, fully open
export LLM_MODEL="Qwen/Qwen2.5-1.5B-Instruct"          # 1.5B, lightweight

# Force a device
export LLM_DEVICE=cuda    # GPU (auto-detected if available)
export LLM_DEVICE=mps     # Mac Apple Silicon
export LLM_DEVICE=cpu     # CPU only (slow but safe)

# Tune generation
export MAX_NEW_TOKENS=300
```

## Run the eval

16 Q&A pairs with ground-truth answers from the press releases:

```bash
python -m eval.run_eval
```

Reports answer accuracy, retrieval accuracy, and per-question latency.

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
    rag.py         retrieval + prompt + SLM generation
    cli.py         typer entrypoint
  eval/
    eval_set.json  16 ground-truth Q&A pairs
    run_eval.py    automated eval runner
```

## Results so far

| Model | Answer accuracy | Retrieval accuracy | Avg latency |
|---|---|---|---|
| Qwen2.5-0.5B (MPS) | 11/16 (69%) | 16/16 (100%) | ~27s |
| SmolLM2-1.7B (CPU) | 1/1 tested (correct) | 1/1 (100%) | ~2h (CPU too slow) |

Retrieval is the strong link. Generator quality scales with model size.
