from __future__ import annotations

import os

from anthropic import Anthropic

from config import LLM_MAX_RETRIES, REQUEST_TIMEOUT
from llm.client import Completion, LLMClient, Message


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None):
        self.provider = "anthropic"
        self.model = model
        self._client = Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"],
            timeout=REQUEST_TIMEOUT,
            max_retries=LLM_MAX_RETRIES,
        )

    def complete(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float | None = 1.0,
    ) -> Completion:
        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []
        for message in messages:
            if message.role == "system":
                system_parts.append(message.content)
            else:
                chat_messages.append({"role": message.role, "content": message.content})

        kwargs: dict = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = self._client.messages.create(**kwargs)
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            content=text,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )


def create_anthropic_client(model: str) -> LLMClient:
    return AnthropicClient(model=model)
