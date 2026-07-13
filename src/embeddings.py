"""Sentence-transformer embeddings with batched encoding."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk


class Encoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", device: str = "cpu"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def encode_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> np.ndarray:
        texts = [c.text for c in chunks]
        print(f"Encoding {len(texts)} chunks (dim={self.dim})")
        return self.encode(texts, batch_size=batch_size)

    def encode_query(self, query: str) -> np.ndarray:
        vec = self.encode([query], show_progress=False)
        return vec[0]
