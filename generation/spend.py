from __future__ import annotations

from llm.usage import SpendCapExceeded, usage

# Approximate USD price per 1M tokens, as (input, output).
# Keyed by substring so versioned model names still match.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "o4-mini": (1.10, 4.40),
    "o3": (2.00, 8.00),
    "gpt-5": (1.25, 10.00),
    "claude-opus": (15.00, 75.00),
    "claude-3-opus": (15.00, 75.00),
    "claude-sonnet": (3.00, 15.00),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-haiku": (0.80, 4.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}

# Conservative fallback for unknown models so the cap still bites.
PRICE_FALLBACK: tuple[float, float] = (10.00, 30.00)


def configure_spend(max_spend: float | None) -> None:
    usage.configure(PRICES, max_spend, PRICE_FALLBACK)


def reset_spend() -> None:
    usage.reset()


def current_spend() -> float:
    return usage.cost()


def spend_snapshot() -> dict:
    return usage.snapshot()


__all__ = [
    "SpendCapExceeded",
    "configure_spend",
    "reset_spend",
    "current_spend",
    "spend_snapshot",
    "PRICES",
    "PRICE_FALLBACK",
]
