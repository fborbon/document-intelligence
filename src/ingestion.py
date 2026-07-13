"""PDF ingestion: text extraction + metadata harvesting."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF


@dataclass
class PageDocument:
    doc_id: str
    source: str          # file path
    page: int
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _doc_id(path: Path) -> str:
    return hashlib.md5(path.name.encode()).hexdigest()[:12]


def extract_pages(pdf_path: Path) -> Iterator[PageDocument]:
    """Yield one PageDocument per page in the PDF."""
    doc = fitz.open(str(pdf_path))
    doc_id = _doc_id(pdf_path)

    meta = doc.metadata or {}
    base_meta = {
        "title": meta.get("title", pdf_path.stem),
        "author": meta.get("author", ""),
        "num_pages": doc.page_count,
        "filename": pdf_path.name,
    }

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if not text:
            continue
        yield PageDocument(
            doc_id=doc_id,
            source=str(pdf_path),
            page=page_num + 1,
            text=text,
            metadata={**base_meta, "page": page_num + 1},
        )

    doc.close()


def load_directory(raw_dir: Path, processed_dir: Path) -> list[PageDocument]:
    """Extract all PDFs in raw_dir and cache results to processed_dir."""
    processed_dir.mkdir(parents=True, exist_ok=True)
    all_docs: list[PageDocument] = []

    pdfs = list(raw_dir.glob("**/*.pdf"))
    print(f"Found {len(pdfs)} PDF(s) in {raw_dir}")

    for pdf_path in pdfs:
        cache_file = processed_dir / f"{_doc_id(pdf_path)}.json"
        if cache_file.exists():
            pages = [PageDocument(**d) for d in json.loads(cache_file.read_text())]
            print(f"  [cache] {pdf_path.name} ({len(pages)} pages)")
        else:
            pages = list(extract_pages(pdf_path))
            cache_file.write_text(json.dumps([p.to_dict() for p in pages], ensure_ascii=False, indent=2))
            print(f"  [new]   {pdf_path.name} ({len(pages)} pages)")

        all_docs.extend(pages)

    return all_docs
