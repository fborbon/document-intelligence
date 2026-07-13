"""Text chunking strategies for document ingestion."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from .ingestion import PageDocument


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source: str
    page: int
    text: str
    chunk_index: int
    strategy: str
    metadata: dict = field(default_factory=dict)


def _make_id(doc_id: str, page: int, idx: int) -> str:
    return f"{doc_id}-p{page}-c{idx}"


def fixed_size_chunks(
    page: PageDocument,
    chunk_size: int = 512,
    overlap: int = 64,
    min_length: int = 50,
) -> list[Chunk]:
    """Split page text into overlapping fixed-size character windows."""
    text = page.text
    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        # extend to nearest sentence boundary if possible
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= min_length:
            chunks.append(
                Chunk(
                    chunk_id=_make_id(page.doc_id, page.page, idx),
                    doc_id=page.doc_id,
                    source=page.source,
                    page=page.page,
                    text=chunk_text,
                    chunk_index=idx,
                    strategy="fixed",
                    metadata=page.metadata,
                )
            )
            idx += 1

        start = end - overlap if end < len(text) else len(text)

    return chunks


def paragraph_chunks(
    page: PageDocument,
    max_size: int = 800,
    min_length: int = 50,
) -> list[Chunk]:
    """Split on double newlines (paragraphs), merging short ones."""
    paragraphs = re.split(r"\n{2,}", page.text)
    merged: list[str] = []
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(buffer) + len(para) < max_size:
            buffer = (buffer + "\n\n" + para).strip() if buffer else para
        else:
            if buffer:
                merged.append(buffer)
            buffer = para

    if buffer:
        merged.append(buffer)

    return [
        Chunk(
            chunk_id=_make_id(page.doc_id, page.page, i),
            doc_id=page.doc_id,
            source=page.source,
            page=page.page,
            text=t,
            chunk_index=i,
            strategy="paragraph",
            metadata=page.metadata,
        )
        for i, t in enumerate(merged)
        if len(t) >= min_length
    ]


def chunk_documents(
    pages: list[PageDocument],
    strategy: Literal["fixed", "paragraph"] = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_length: int = 50,
) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for page in pages:
        if strategy == "fixed":
            all_chunks.extend(
                fixed_size_chunks(page, chunk_size, chunk_overlap, min_chunk_length)
            )
        else:
            all_chunks.extend(paragraph_chunks(page, chunk_size, min_chunk_length))
    print(f"Chunking: {len(pages)} pages → {len(all_chunks)} chunks ({strategy})")
    return all_chunks
