"""FastAPI serving layer for the document intelligence pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.embeddings import Encoder
from src.vectorstore import VectorStore
from src.rag import RAGPipeline
from src.extractor import StructuredExtractor

ROOT = Path(__file__).parent
cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

app = FastAPI(
    title="Document Intelligence API",
    description="RAG pipeline over technical and regulatory PDF documents",
    version="1.0.0",
)

# Lazy globals — loaded on first request
_encoder: Encoder | None = None
_store: VectorStore | None = None
_pipeline: RAGPipeline | None = None
_extractor: StructuredExtractor | None = None


def _load():
    global _encoder, _store, _pipeline, _extractor
    if _pipeline is not None:
        return
    store_dir = ROOT / cfg["data"]["vectorstore_dir"]
    if not (store_dir / "index.faiss").exists():
        raise RuntimeError("Index not built. Run: python scripts/build_index.py")
    _encoder = Encoder(cfg["embeddings"]["model"], cfg["embeddings"]["device"])
    _store = VectorStore.load(store_dir)
    _pipeline = RAGPipeline(
        vector_store=_store,
        encoder=_encoder,
        model=cfg["extraction"]["llm_model"],
        top_k=cfg["retrieval"]["top_k"],
        score_threshold=cfg["retrieval"]["score_threshold"],
    )
    _extractor = StructuredExtractor(model=cfg["extraction"]["llm_model"])


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[dict]
    latency_ms: float
    tokens_used: int


class ExtractRequest(BaseModel):
    query: str
    mode: str = "technical"  # "technical" | "regulatory"


@app.on_event("startup")
async def startup():
    try:
        _load()
    except RuntimeError as e:
        print(f"[warn] {e} — /query and /extract will fail until index is built.")


@app.get("/health")
def health() -> dict:
    indexed = _store.index.ntotal if _store else 0
    return {"status": "ok", "indexed_chunks": indexed}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    _load()
    result = _pipeline.query(req.question, track=False)
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            {"title": r["metadata"].get("title", r["source"]), "page": r["page"], "score": r["score"]}
            for r in result.retrieved
        ],
        latency_ms=result.latency_ms,
        tokens_used=result.input_tokens + result.output_tokens,
    )


@app.post("/extract")
def extract(req: ExtractRequest) -> dict[str, Any]:
    _load()
    query_vec = _encoder.encode_query(req.query)
    retrieved = _store.search(query_vec, top_k=cfg["retrieval"]["top_k"])
    if req.mode == "regulatory":
        return _extractor.extract_regulatory_rules(retrieved)
    return _extractor.extract_technical_details(retrieved)


@app.get("/stats")
def stats() -> dict:
    _load()
    sources = {}
    for chunk in _store.chunks:
        title = chunk.metadata.get("title", chunk.source)
        sources[title] = sources.get(title, 0) + 1
    return {
        "total_chunks": _store.index.ntotal,
        "embedding_dim": _store.dim,
        "documents": [{"title": t, "chunks": n} for t, n in sorted(sources.items())],
    }
