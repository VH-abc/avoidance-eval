from __future__ import annotations

import os

from openai import OpenAI

from config import LLM_MAX_RETRIES, REQUEST_TIMEOUT
from llm.client import Completion, LLMClient, Message


class OpenAIClient:
    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        self.provider = "openai"
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.environ["OPENAI_API_KEY"],
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=REQUEST_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )

    def complete(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float | None = 1.0,
    ) -> Completion:
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage
        return Completion(
            content=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )


def create_openai_client(model: str) -> LLMClient:
    return OpenAIClient(model=model)
