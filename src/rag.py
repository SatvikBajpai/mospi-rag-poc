"""Retrieval + SLM generation."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import chromadb
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import COLLECTION, DB_DIR, EMBED_MODEL, LLM_DEVICE, LLM_MODEL, MAX_NEW_TOKENS, TOP_K

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


@lru_cache(maxsize=1)
def _llm():
    if LLM_DEVICE:
        device = LLM_DEVICE
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    dtype = torch.float32 if device == "cpu" else torch.float16
    print(f"[rag] loading LLM {LLM_MODEL} on {device}")
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForCausalLM.from_pretrained(LLM_MODEL, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return tok, model, device


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
    tok, model, device = _llm()
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


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
