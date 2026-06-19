from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


class SpendCapExceeded(Exception):
    """Raised when the configured spend cap is exceeded mid-run."""


@dataclass
class _ModelUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0


@dataclass
class UsageMonitor:
    """Process-wide token-usage accumulator with optional pricing + spend cap.

    The accumulator lives in the llm layer so the client wrapper can record
    usage without importing higher-level packages. Pricing and the cap are
    injected via `configure` (typically from generation/spend.py).
    """

    _lock: Lock = field(default_factory=Lock, repr=False)
    _by_model: dict[str, _ModelUsage] = field(default_factory=dict)
    _prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    _price_fallback: tuple[float, float] | None = None
    _max_spend: float | None = None

    def configure(
        self,
        prices: dict[str, tuple[float, float]],
        max_spend: float | None,
        price_fallback: tuple[float, float] | None = None,
    ) -> None:
        with self._lock:
            self._prices = dict(prices)
            self._max_spend = max_spend
            self._price_fallback = price_fallback

    def reset(self) -> None:
        with self._lock:
            self._by_model = {}

    def record(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            entry = self._by_model.setdefault(model, _ModelUsage())
            entry.prompt_tokens += int(prompt_tokens or 0)
            entry.completion_tokens += int(completion_tokens or 0)
            entry.calls += 1
            cost = self._cost_locked()
            cap = self._max_spend
        if cap is not None and cost > cap:
            raise SpendCapExceeded(
                f"Spend cap exceeded: ${cost:.2f} > ${cap:.2f}. "
                "Stopping to avoid further API charges."
            )

    def _price_for(self, model: str) -> tuple[float, float] | None:
        if model in self._prices:
            return self._prices[model]
        for key, price in self._prices.items():
            if key in model or model in key:
                return price
        return self._price_fallback

    def _cost_locked(self) -> float:
        total = 0.0
        for model, entry in self._by_model.items():
            price = self._price_for(model)
            if price is None:
                continue
            in_price, out_price = price
            total += entry.prompt_tokens / 1_000_000 * in_price
            total += entry.completion_tokens / 1_000_000 * out_price
        return total

    def cost(self) -> float:
        with self._lock:
            return self._cost_locked()

    def snapshot(self) -> dict:
        with self._lock:
            models = {
                model: {
                    "prompt_tokens": entry.prompt_tokens,
                    "completion_tokens": entry.completion_tokens,
                    "calls": entry.calls,
                    "cost": (
                        (entry.prompt_tokens / 1_000_000 * self._price_for(model)[0])
                        + (entry.completion_tokens / 1_000_000 * self._price_for(model)[1])
                        if self._price_for(model) is not None
                        else None
                    ),
                }
                for model, entry in self._by_model.items()
            }
            return {
                "total_cost": self._cost_locked(),
                "max_spend": self._max_spend,
                "by_model": models,
            }


usage = UsageMonitor()
