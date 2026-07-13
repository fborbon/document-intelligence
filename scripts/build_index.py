#!/usr/bin/env python3
"""
Ingest all PDFs in data/raw/ → chunk → embed → build FAISS index.
Tracks the run with MLflow.
"""
import sys
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.ingestion import load_directory
from src.chunking import chunk_documents
from src.embeddings import Encoder
from src.vectorstore import VectorStore


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

    raw_dir       = ROOT / cfg["data"]["raw_dir"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    store_dir     = ROOT / cfg["data"]["vectorstore_dir"]

    tracking_uri = cfg["mlflow"]["tracking_uri"]
    if not tracking_uri.startswith("sqlite:///"):
        tracking_uri = f"sqlite:///{ROOT / tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="build_index"):
        # -- Ingestion
        pages = load_directory(raw_dir, processed_dir)
        if not pages:
            print("No pages extracted. Run scripts/download_data.py first.")
            sys.exit(1)

        # -- Chunking
        chunks = chunk_documents(
            pages,
            strategy="fixed",
            chunk_size=cfg["ingestion"]["chunk_size"],
            chunk_overlap=cfg["ingestion"]["chunk_overlap"],
            min_chunk_length=cfg["ingestion"]["min_chunk_length"],
        )

        # -- Embeddings
        encoder = Encoder(
            model_name=cfg["embeddings"]["model"],
            device=cfg["embeddings"]["device"],
        )
        embeddings = encoder.encode_chunks(chunks, batch_size=cfg["embeddings"]["batch_size"])

        # -- Vector store
        store = VectorStore(dim=encoder.dim)
        store.add(chunks, embeddings)
        store.save(store_dir)

        # -- MLflow logging
        mlflow.log_params({
            "embedding_model": cfg["embeddings"]["model"],
            "chunk_size": cfg["ingestion"]["chunk_size"],
            "chunk_overlap": cfg["ingestion"]["chunk_overlap"],
            "chunking_strategy": "fixed",
        })
        mlflow.log_metrics({
            "num_pdfs": len({p.source for p in pages}),
            "num_pages": len(pages),
            "num_chunks": len(chunks),
            "embedding_dim": encoder.dim,
        })

        print(f"\nIndex built: {len(chunks)} chunks, dim={encoder.dim}")
        print(f"Stored at: {store_dir}")


if __name__ == "__main__":
    main()
