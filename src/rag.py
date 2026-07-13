"""RAG pipeline: retrieve + generate answer with MLflow tracking."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import anthropic
import mlflow

RAG_SYSTEM_PROMPT = """\
You are a document intelligence assistant specialized in technical and regulatory documents.
Answer questions using ONLY the provided context. Cite the source document and page number for each claim.
If the answer cannot be found in the context, say "Not found in the indexed documents."
"""

RAG_USER_PROMPT = """\
Context (retrieved by semantic search):
{context}

Question: {question}

Provide a precise, well-structured answer with citations like [Source, p.N].
"""


@dataclass
class RAGResult:
    question: str
    answer: str
    retrieved: list[dict]
    latency_ms: float
    model: str
    input_tokens: int
    output_tokens: int


class RAGPipeline:
    def __init__(
        self,
        vector_store,
        encoder,
        model: str = "claude-sonnet-4-6",
        top_k: int = 5,
        score_threshold: float = 0.3,
        max_tokens: int = 2048,
    ):
        self.store = vector_store
        self.encoder = encoder
        self.client = anthropic.Anthropic()
        self.model = model
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.max_tokens = max_tokens

    def _format_context(self, retrieved: list[dict]) -> str:
        parts = []
        for r in retrieved:
            title = r["metadata"].get("title", r["source"])
            parts.append(f"[{title}, p.{r['page']}]\n{r['text']}")
        return "\n\n---\n\n".join(parts)

    def query(self, question: str, track: bool = True) -> RAGResult:
        t0 = time.time()

        query_vec = self.encoder.encode_query(question)
        retrieved = self.store.search(query_vec, top_k=self.top_k)
        retrieved = [r for r in retrieved if r["score"] >= self.score_threshold]

        context = self._format_context(retrieved) if retrieved else "No relevant documents found."

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=RAG_SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": RAG_USER_PROMPT.format(context=context, question=question),
                }
            ],
        )

        latency_ms = (time.time() - t0) * 1000
        result = RAGResult(
            question=question,
            answer=response.content[0].text,
            retrieved=retrieved,
            latency_ms=latency_ms,
            model=self.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        if track:
            self._log_to_mlflow(result)

        return result

    def _log_to_mlflow(self, result: RAGResult) -> None:
        try:
            with mlflow.start_run(run_name="rag_query", nested=True):
                mlflow.log_params({
                    "model": result.model,
                    "top_k": self.top_k,
                    "score_threshold": self.score_threshold,
                })
                mlflow.log_metrics({
                    "latency_ms": result.latency_ms,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "retrieved_chunks": len(result.retrieved),
                    "avg_retrieval_score": (
                        sum(r["score"] for r in result.retrieved) / len(result.retrieved)
                        if result.retrieved else 0.0
                    ),
                })
                mlflow.log_text(result.question, "question.txt")
                mlflow.log_text(result.answer, "answer.txt")
        except Exception:
            pass  # MLflow tracking is best-effort
