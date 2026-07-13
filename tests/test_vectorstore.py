"""Unit tests for VectorStore (no network, no LLM)."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.chunking import Chunk
from src.vectorstore import VectorStore


def _random_chunks(n: int, dim: int = 384) -> tuple[list[Chunk], np.ndarray]:
    chunks = [
        Chunk(
            chunk_id=f"c{i}",
            doc_id="doc1",
            source="test.pdf",
            page=i + 1,
            text=f"Sample text chunk {i}",
            chunk_index=i,
            strategy="fixed",
        )
        for i in range(n)
    ]
    vecs = np.random.rand(n, dim).astype(np.float32)
    # normalize to unit vectors for cosine sim
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return chunks, vecs


def test_add_and_search():
    chunks, vecs = _random_chunks(20)
    store = VectorStore(dim=384)
    store.add(chunks, vecs)
    assert store.index.ntotal == 20

    results = store.search(vecs[0], top_k=3)
    assert len(results) == 3
    # top result should be the query vector itself
    assert results[0]["chunk_id"] == "c0"
    assert results[0]["score"] > 0.99


def test_save_and_load():
    chunks, vecs = _random_chunks(10)
    store = VectorStore(dim=384)
    store.add(chunks, vecs)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store.save(tmp_path)

        loaded = VectorStore.load(tmp_path)
        assert loaded.index.ntotal == 10
        assert len(loaded.chunks) == 10

        results = loaded.search(vecs[5], top_k=1)
        assert results[0]["chunk_id"] == "c5"


def test_empty_store():
    store = VectorStore(dim=384)
    assert store.is_empty


if __name__ == "__main__":
    test_add_and_search()
    test_save_and_load()
    test_empty_store()
    print("All vectorstore tests passed.")
