"""FAISS-backed vector store with chunk metadata persistence."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np

from .chunking import Chunk


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)   # inner product on unit vecs = cosine sim
        self.chunks: list[Chunk] = []

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert embeddings.shape == (len(chunks), self.dim), "Embedding shape mismatch"
        self.index.add(embeddings.astype(np.float32))
        self.chunks.extend(chunks)
        print(f"VectorStore: {self.index.ntotal} total vectors")

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        scores, indices = self.index.search(
            query_vec.reshape(1, -1).astype(np.float32), top_k
        )
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append(
                {
                    "score": float(score),
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )
        return results

    def save(self, store_dir: Path) -> None:
        store_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(store_dir / "index.faiss"))
        with open(store_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        (store_dir / "meta.json").write_text(
            json.dumps({"dim": self.dim, "ntotal": self.index.ntotal})
        )
        print(f"VectorStore saved to {store_dir}")

    @classmethod
    def load(cls, store_dir: Path) -> "VectorStore":
        meta = json.loads((store_dir / "meta.json").read_text())
        store = cls(dim=meta["dim"])
        store.index = faiss.read_index(str(store_dir / "index.faiss"))
        with open(store_dir / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        print(f"VectorStore loaded: {store.index.ntotal} vectors (dim={store.dim})")
        return store

    @property
    def is_empty(self) -> bool:
        return self.index.ntotal == 0
