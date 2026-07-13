"""Structured information extraction from retrieved chunks using Claude."""
from __future__ import annotations

import json
from typing import Any

import anthropic


EXTRACTION_PROMPT = """\
You are a document intelligence assistant. Given the retrieved context from technical/regulatory PDFs, extract structured information.

Context chunks (ranked by relevance):
{context}

Task: {task}

Return a JSON object with the extracted information. Be precise and only extract what is explicitly stated in the context. If information is not found, use null.
"""

REGULATORY_EXTRACTION_SCHEMA = {
    "rules": "list of extracted regulatory rules or requirements",
    "entities": "organizations, standards bodies, or systems mentioned",
    "obligations": "mandatory requirements (MUST, SHALL, REQUIRED)",
    "dates_deadlines": "any dates, deadlines, or effective dates",
    "penalties": "consequences of non-compliance if mentioned",
}

TECHNICAL_EXTRACTION_SCHEMA = {
    "methods": "algorithms, techniques, or methods described",
    "datasets": "datasets used or referenced",
    "metrics": "evaluation metrics and results",
    "architecture": "system or model architecture details",
    "limitations": "stated limitations or future work",
}


class StructuredExtractor:
    def __init__(self, model: str = "claude-sonnet-4-6", max_tokens: int = 2048):
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def _format_context(self, retrieved: list[dict]) -> str:
        lines = []
        for i, r in enumerate(retrieved, 1):
            title = r["metadata"].get("title", r["source"])
            lines.append(f"[{i}] Source: {title}, Page {r['page']} (score={r['score']:.3f})")
            lines.append(r["text"])
            lines.append("")
        return "\n".join(lines)

    def extract(self, retrieved: list[dict], task: str) -> dict[str, Any]:
        context = self._format_context(retrieved)
        prompt = EXTRACTION_PROMPT.format(context=context, task=task)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        # strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw_response": raw}

    def extract_regulatory_rules(self, retrieved: list[dict]) -> dict[str, Any]:
        task = (
            f"Extract regulatory rules and compliance requirements. "
            f"Expected schema: {json.dumps(REGULATORY_EXTRACTION_SCHEMA, indent=2)}"
        )
        return self.extract(retrieved, task)

    def extract_technical_details(self, retrieved: list[dict]) -> dict[str, Any]:
        task = (
            f"Extract technical details about methods, models, and results. "
            f"Expected schema: {json.dumps(TECHNICAL_EXTRACTION_SCHEMA, indent=2)}"
        )
        return self.extract(retrieved, task)
