from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import boto3

logger = logging.getLogger(__name__)


class BedrockProvider:
    def __init__(
        self,
        region: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self._region = region or os.getenv("AWS_REGION", "us-east-1")
        self._model_id = model_id or os.getenv(
            "CLAUDE_MODEL", "anthropic.claude-sonnet-4-20250514"
        )
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self._region,
        )

    async def invoke(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> str:
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = await asyncio.to_thread(
            self._client.invoke_model,
            modelId=self._model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )

        result = json.loads(response["body"].read())
        text_blocks = [
            block["text"]
            for block in result.get("content", [])
            if block.get("type") == "text"
        ]
        return "\n".join(text_blocks)
