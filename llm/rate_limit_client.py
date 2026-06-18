from __future__ import annotations

import time

from llm.client import Completion, LLMClient, Message
from llm.rate_limits import SLOW_REQUEST_SECONDS, is_rate_limit_error, monitor


class RateLimitAwareClient:
    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.provider = inner.provider
        self.model = inner.model

    def complete(
        self,
        messages: list[Message],
        max_tokens: int,
        temperature: float = 1.0,
    ) -> Completion:
        start = time.monotonic()
        try:
            completion = self._inner.complete(messages, max_tokens, temperature)
        except Exception as exc:
            if is_rate_limit_error(exc):
                monitor.record_hit(self.provider, self.model, exc)
            raise
        elapsed = time.monotonic() - start
        if elapsed >= SLOW_REQUEST_SECONDS:
            monitor.record_slow(self.provider, self.model, elapsed)
        return completion
