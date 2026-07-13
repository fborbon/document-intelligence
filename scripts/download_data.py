#!/usr/bin/env python3
"""
Download real-world PDF technical documents for the document-intelligence demo.

Sources:
  1. arXiv papers on RAG, document understanding, and NLP (via arxiv API)
  2. SEC EDGAR 10-K annual report excerpts (via public API, no auth)
"""
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
(RAW_DIR / "arxiv").mkdir(parents=True, exist_ok=True)
(RAW_DIR / "sec").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  arXiv papers
# ---------------------------------------------------------------------------
ARXIV_PAPERS = [
    # (arxiv_id, short_title)
    ("2005.11401", "rag-lewis-2020"),            # Original RAG paper
    ("2312.10997", "rag-survey-gao-2023"),        # RAG survey
    ("2103.00020", "clip-radford-2021"),          # CLIP (multimodal doc understanding)
    ("1904.01038", "layoutlm-xu-2019"),           # LayoutLM for document understanding
    ("2204.02311", "docformer-appalaraju-2021"),  # DocFormer
]


def download_arxiv(paper_id: str, name: str) -> Path:
    out = RAW_DIR / "arxiv" / f"{name}.pdf"
    if out.exists():
        print(f"  [skip] {name}.pdf already exists")
        return out
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    print(f"  Downloading arXiv:{paper_id} → {name}.pdf")
    try:
        urllib.request.urlretrieve(url, out)
        time.sleep(1)  # polite delay
    except Exception as e:
        print(f"  [warn] Failed to download {paper_id}: {e}")
    return out


# ---------------------------------------------------------------------------
# 2.  SEC EDGAR – download 10-K filing for a public company
#     Microsoft (CIK 0000789019) 2023 annual report
# ---------------------------------------------------------------------------
SEC_FILINGS = [
    {
        "company": "Microsoft",
        "cik": "0000789019",
        "accession": "0000950170-23-035122",
        "document": "msft-20230630.htm",
        "filename": "microsoft-10k-2023.htm",
    },
]


def download_sec_filing(filing: dict) -> Path:
    """Download SEC filing HTML (converted to PDF via print if needed)."""
    out = RAW_DIR / "sec" / filing["filename"]
    if out.exists():
        print(f"  [skip] {filing['filename']} already exists")
        return out

    acc_clean = filing["accession"].replace("-", "")
    url = (
        f"https://www.sec.gov/Archives/edgar/full-index/"
        f"{filing['cik']}/{acc_clean}/{filing['document']}"
    )
    # Use the EDGAR viewer API for a cleaner fetch
    alt_url = (
        f"https://efts.sec.gov/LATEST/search-index?q=%22{filing['company']}%22"
        f"&dateRange=custom&startdt=2023-01-01&enddt=2023-12-31&forms=10-K"
    )

    print(f"  Downloading SEC 10-K: {filing['company']} → {filing['filename']}")
    headers = {"User-Agent": "document-intelligence-demo admin@example.com"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            content = r.read()
        out.write_bytes(content)
        print(f"  Saved {len(content)//1024} KB")
    except Exception as e:
        print(f"  [warn] SEC fetch failed: {e}")
    return out


# ---------------------------------------------------------------------------
# 3.  Supplementary: NIST SP 800-53 (public regulatory document, PDF)
#     Published by US federal government, freely available
# ---------------------------------------------------------------------------
PUBLIC_PDFS = [
    {
        "url": "https://nvlpubs.nist.gov/nistpubs/SpecialPublications/NIST.SP.800-53r5.pdf",
        "filename": "nist-sp800-53r5-security-controls.pdf",
        "subdir": "regulatory",
        "description": "NIST SP 800-53 Rev5 – Security and Privacy Controls (regulatory reference)",
    },
]


def download_public_pdf(entry: dict) -> Path:
    subdir = RAW_DIR / entry["subdir"]
    subdir.mkdir(parents=True, exist_ok=True)
    out = subdir / entry["filename"]
    if out.exists():
        print(f"  [skip] {entry['filename']} already exists")
        return out
    print(f"  Downloading {entry['description']}")
    headers = {"User-Agent": "document-intelligence-demo admin@example.com"}
    req = urllib.request.Request(entry["url"], headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            content = r.read()
        out.write_bytes(content)
        print(f"  Saved {len(content)//1024} KB → {out.name}")
    except Exception as e:
        print(f"  [warn] Failed: {e}")
    return out


def main():
    print("=== Downloading document-intelligence demo data ===\n")

    print("[1/3] arXiv technical papers:")
    for paper_id, name in ARXIV_PAPERS:
        download_arxiv(paper_id, name)

    print("\n[2/3] Public regulatory PDFs:")
    for entry in PUBLIC_PDFS:
        download_public_pdf(entry)

    print("\n[3/3] SEC EDGAR filings (HTML format):")
    for filing in SEC_FILINGS:
        download_sec_filing(filing)

    print("\nDone. Files saved to data/raw/")
    total = list(RAW_DIR.rglob("*.*"))
    for f in sorted(total):
        size = f.stat().st_size // 1024
        print(f"  {f.relative_to(RAW_DIR)}  ({size} KB)")


if __name__ == "__main__":
    main()
