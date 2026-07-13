"""Unit tests for chunking strategies."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingestion import PageDocument
from src.chunking import fixed_size_chunks, paragraph_chunks, chunk_documents


def _make_page(text: str, doc_id: str = "test01", page: int = 1) -> PageDocument:
    return PageDocument(doc_id=doc_id, source="test.pdf", page=page, text=text)


def test_fixed_chunks_basic():
    page = _make_page("A" * 1200)
    chunks = fixed_size_chunks(page, chunk_size=512, overlap=64, min_length=10)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.text) >= 10
        assert c.doc_id == "test01"
        assert c.strategy == "fixed"


def test_fixed_chunks_short_text():
    page = _make_page("Short text.")
    chunks = fixed_size_chunks(page, chunk_size=512, overlap=64, min_length=5)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."


def test_paragraph_chunks():
    # Each paragraph ~22 chars; max_size=40 < 22+22 forces splits
    text = "First paragraph here.\n\nSecond paragraph too.\n\nThird paragraph end."
    page = _make_page(text)
    chunks = paragraph_chunks(page, max_size=40, min_length=5)
    assert len(chunks) == 3
    assert chunks[0].strategy == "paragraph"


def test_paragraph_chunks_merging():
    text = "Short.\n\nAlso short.\n\nAnd another short one here."
    page = _make_page(text)
    chunks = paragraph_chunks(page, max_size=200, min_length=5)
    # All three should merge into one or two chunks
    assert len(chunks) <= 2


def test_chunk_ids_unique():
    page = _make_page("X" * 2000)
    chunks = fixed_size_chunks(page, chunk_size=300, overlap=50, min_length=10)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "Chunk IDs must be unique"


def test_chunk_documents_aggregates():
    pages = [
        _make_page("Page one content " * 30, doc_id="doc1", page=1),
        _make_page("Page two content " * 30, doc_id="doc1", page=2),
    ]
    chunks = chunk_documents(pages, strategy="fixed", chunk_size=256, chunk_overlap=32)
    assert len(chunks) > 2
    doc_ids = {c.doc_id for c in chunks}
    assert doc_ids == {"doc1"}


if __name__ == "__main__":
    test_fixed_chunks_basic()
    test_fixed_chunks_short_text()
    test_paragraph_chunks()
    test_paragraph_chunks_merging()
    test_chunk_ids_unique()
    test_chunk_documents_aggregates()
    print("All chunking tests passed.")
