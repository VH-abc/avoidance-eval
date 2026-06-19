from __future__ import annotations

from llm.client import Message
from llm.factory import create_client


def probe_model(provider: str, model: str) -> str:
    """Make a tiny call to confirm a model string is reachable.

    Returns the model's reply on success. Any failure (e.g. an invalid model
    string) raises, which intentionally stops the run before larger spend.
    """
    client = create_client(provider, model)
    completion = client.complete(
        messages=[Message(role="user", content="Reply with the single word: ok")],
        max_tokens=5,
        temperature=None,
    )
    return completion.content.strip()


if __name__ == "__main__":
    import sys

    from config import GENERATION_MODEL, GENERATION_PROVIDER

    provider = sys.argv[1] if len(sys.argv) > 1 else GENERATION_PROVIDER
    model = sys.argv[2] if len(sys.argv) > 2 else GENERATION_MODEL
    print(f"Probing provider={provider} model={model} ...")
    reply = probe_model(provider, model)
    print(f"OK -> {reply!r}")
