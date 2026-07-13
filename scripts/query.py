#!/usr/bin/env python3
"""
Interactive RAG query CLI.

Usage:
  python scripts/query.py "What is retrieval augmented generation?"
  python scripts/query.py --extract-rules "What are the mandatory access control requirements?"
  python scripts/query.py --interactive
"""
import argparse
import sys
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.embeddings import Encoder
from src.vectorstore import VectorStore
from src.rag import RAGPipeline
from src.extractor import StructuredExtractor


def load_pipeline(cfg: dict):
    store_dir = ROOT / cfg["data"]["vectorstore_dir"]
    if not (store_dir / "index.faiss").exists():
        print("Index not found. Run: python scripts/build_index.py")
        sys.exit(1)

    encoder = Encoder(
        model_name=cfg["embeddings"]["model"],
        device=cfg["embeddings"]["device"],
    )
    store = VectorStore.load(store_dir)
    pipeline = RAGPipeline(
        vector_store=store,
        encoder=encoder,
        model=cfg["extraction"]["llm_model"],
        top_k=cfg["retrieval"]["top_k"],
        score_threshold=cfg["retrieval"]["score_threshold"],
        max_tokens=cfg["extraction"]["max_tokens"],
    )
    return pipeline, store, encoder


def print_result(result) -> None:
    print("\n" + "=" * 70)
    print(f"Question: {result.question}")
    print("=" * 70)
    print(result.answer)
    print("-" * 70)
    print(f"Retrieved {len(result.retrieved)} chunks | "
          f"Latency: {result.latency_ms:.0f}ms | "
          f"Tokens: {result.input_tokens}+{result.output_tokens}")
    if result.retrieved:
        print("\nSources:")
        for r in result.retrieved:
            title = r["metadata"].get("title", r["source"])
            print(f"  • {title}, p.{r['page']} (score={r['score']:.3f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Document Intelligence RAG Query CLI")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--extract-rules", action="store_true", help="Run structured regulatory extraction")
    parser.add_argument("--extract-tech", action="store_true", help="Run structured technical extraction")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    if not tracking_uri.startswith("sqlite:///"):
        tracking_uri = f"sqlite:///{ROOT / tracking_uri}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    pipeline, store, encoder = load_pipeline(cfg)

    if args.interactive:
        print("Document Intelligence RAG — Interactive Mode (Ctrl+C to exit)\n")
        while True:
            try:
                question = input("Question: ").strip()
                if not question:
                    continue
                result = pipeline.query(question)
                print_result(result)
            except KeyboardInterrupt:
                print("\nBye.")
                break

    elif args.question:
        if args.extract_rules or args.extract_tech:
            query_vec = encoder.encode_query(args.question)
            retrieved = store.search(query_vec, top_k=cfg["retrieval"]["top_k"])
            extractor = StructuredExtractor(model=cfg["extraction"]["llm_model"])
            if args.extract_rules:
                result = extractor.extract_regulatory_rules(retrieved)
            else:
                result = extractor.extract_technical_details(retrieved)
            import json
            print(json.dumps(result, indent=2))
        else:
            result = pipeline.query(args.question)
            print_result(result)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
