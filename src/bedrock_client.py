"""Drop-in replacement for `anthropic.Anthropic`, backed by Amazon Nova on AWS Bedrock instead
of the Anthropic API — the Anthropic Console key this repo used is shared across several other
projects and ran out of credit. Auth is via ambient AWS credentials (IAM role, or `aws configure`
for local dev) — no API key needed, billed through AWS instead.

Named `Anthropic` and exposing the same `.messages.create(model=, max_tokens=, messages=,
system=, temperature=)` surface as the real SDK — including a `.usage.input_tokens/
.output_tokens`-shaped response for the MLflow tracking in rag.py — so call sites only need to
change their import, not their logic.
"""

from __future__ import annotations

import json

NOVA_MODEL_ID = "eu.amazon.nova-lite-v1:0"
BEDROCK_REGION = "eu-west-1"


class _ContentBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _Usage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Message:
    def __init__(self, text: str, stop_reason: str, usage: _Usage) -> None:
        self.content = [_ContentBlock(text)]
        self.stop_reason = stop_reason
        self.usage = usage


class Anthropic:
    def __init__(self, model_id: str = NOVA_MODEL_ID, region: str = BEDROCK_REGION) -> None:
        self._model_id = model_id
        self._region = region
        self._bedrock = None

    @property
    def messages(self) -> Anthropic:
        return self

    def create(
        self,
        model: str,
        max_tokens: int,
        messages: list[dict],
        system: str | None = None,
        temperature: float | None = None,
        **_ignored,
    ) -> _Message:
        if self._bedrock is None:
            import boto3
            self._bedrock = boto3.client("bedrock-runtime", region_name=self._region)

        body: dict = {
            "messages": [
                {"role": m["role"], "content": [{"text": m["content"]}]} for m in messages
            ],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if system:
            body["system"] = [{"text": system}]
        if temperature is not None:
            body["inferenceConfig"]["temperature"] = temperature

        response = self._bedrock.invoke_model(
            modelId=self._model_id, body=json.dumps(body),
            contentType="application/json", accept="application/json",
        )
        result = json.loads(response["body"].read())
        usage = result.get("usage", {})
        return _Message(
            text=result["output"]["message"]["content"][0]["text"],
            stop_reason=result.get("stopReason", ""),
            usage=_Usage(usage.get("inputTokens", 0), usage.get("outputTokens", 0)),
        )
