from __future__ import annotations

from llm.anthropic_client import create_anthropic_client
from llm.client import LLMClient
from llm.openai_client import create_openai_client


def create_client(provider: str, model: str) -> LLMClient:
    if provider == "openai":
        return create_openai_client(model)
    if provider == "anthropic":
        return create_anthropic_client(model)
    raise ValueError(f"Unknown provider: {provider}")
