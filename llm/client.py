from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Completion:
    content: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMClient(Protocol):
    provider: str
    model: str

    def complete(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float | None = 1.0,
    ) -> Completion: ...
