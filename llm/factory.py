from __future__ import annotations

from llm.anthropic_client import create_anthropic_client
from llm.client import LLMClient
from llm.openai_client import create_openai_client
from llm.rate_limit_client import RateLimitAwareClient


def create_client(provider: str, model: str) -> LLMClient:
    if provider == "openai":
        inner = create_openai_client(model)
    elif provider == "anthropic":
        inner = create_anthropic_client(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    return RateLimitAwareClient(inner)
