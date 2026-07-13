# Document Intelligence Pipeline

RAG-based system for querying and extracting structured information from technical and regulatory PDF documents. Demonstrates the core capabilities required for the N-iX Senior ML Engineer role: LLM pipelines, RAG architecture, structured extraction, MLOps tracking, and productionized serving.

## Architecture

```
PDFs (arXiv / NIST / SEC)
        │
        ▼
  src/ingestion.py       ← PyMuPDF text extraction + metadata + caching
        │
        ▼
  src/chunking.py        ← Fixed-size + paragraph chunking strategies
        │
        ▼
  src/embeddings.py      ← sentence-transformers (all-MiniLM-L6-v2)
        │
        ▼
  src/vectorstore.py     ← FAISS IndexFlatIP (cosine similarity)
        │
        ├──▶ src/rag.py          ← Claude LLM answer generation + MLflow tracking
        └──▶ src/extractor.py    ← Structured JSON extraction (regulatory/technical)
                │
                ▼
            api.py               ← FastAPI REST endpoints
```

## Dataset (real-world PDFs)

| File | Domain | Pages |
|------|--------|-------|
| `rag-lewis-2020.pdf` | RAG original paper (arXiv:2005.11401) | 19 |
| `rag-survey-gao-2023.pdf` | RAG survey (arXiv:2312.10997) | 21 |
| `layoutlm-xu-2019.pdf` | Document understanding (arXiv:1904.01038) | 6 |
| `docformer-appalaraju-2021.pdf` | DocFormer (arXiv:2204.02311) | 87 |
| `clip-radford-2021.pdf` | CLIP multimodal (arXiv:2103.00020) | 48 |
| `nist-sp800-53r5-security-controls.pdf` | NIST regulatory controls | 492 |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

# Download PDFs
python3 scripts/download_data.py

# Build FAISS index (runs MLflow experiment)
python3 scripts/build_index.py
```

## Usage

### CLI Query
```bash
# Semantic RAG query
python3 scripts/query.py "How does retrieval augmented generation work?"

# Interactive mode
python3 scripts/query.py --interactive

# Structured extraction — regulatory
python3 scripts/query.py --extract-rules "What are the mandatory access control requirements?"

# Structured extraction — technical
python3 scripts/query.py --extract-tech "What datasets were used for evaluation?"
```

### REST API
```bash
uvicorn api:app --reload

# Health check
curl http://localhost:8000/health

# RAG query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is dense passage retrieval?"}'

# Structured extraction
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "access control requirements", "mode": "regulatory"}'

# Index stats
curl http://localhost:8000/stats
```

### MLflow Tracking
```bash
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
# Open http://localhost:5000
```

### Docker
```bash
docker build -t document-intelligence .
docker run -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY -p 8000:8000 document-intelligence
```

## Tests

```bash
python3 -m pytest tests/ -v
```

## Key Design Decisions

- **Chunking**: Fixed-size with sentence-boundary snapping prevents mid-sentence splits; paragraph mode available for structured docs
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim) balances quality and CPU inference speed; swap for `all-mpnet-base-v2` for higher accuracy
- **Vector store**: FAISS `IndexFlatIP` on L2-normalized vectors = exact cosine similarity; upgrade to `IndexIVFFlat` for >1M chunks
- **LLM**: Claude (`claude-sonnet-4-6`) for answer generation and structured extraction; `temperature=0` for deterministic extraction
- **MLflow**: Tracks per-query latency, token usage, retrieval score, and chunking hyperparameters for drift monitoring
- **Caching**: Page-level JSON cache in `data/processed/` avoids re-parsing unchanged PDFs
