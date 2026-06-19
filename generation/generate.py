from __future__ import annotations

import json
import re

from config import GENERATION_MODEL, GENERATION_PROVIDER
from generation.prompts import DOMAINS, RELATION_TYPES, build_generation_messages
from llm.client import LLMClient
from llm.factory import create_client
from models import QuestionPair

REQUIRED_KEYS = {"x", "y", "answer_x", "answer_y", "relation", "y_proximity_signals"}


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _extract_array(text: str) -> str | None:
    """Return the substring of the first balanced top-level JSON array, or None.

    Pure scan (no exceptions); respects string literals and escapes.
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_candidates(text: str) -> list[dict]:
    """Parse a JSON array of candidate objects from model output.

    LLM output is untrusted and occasionally malformed; this boundary tolerates
    a bad batch (prints a warning and returns what it can) rather than aborting
    an expensive multi-batch run.
    """
    payload = _extract_array(_strip_fences(text))
    if payload is None:
        print("  [generate] warning: no JSON array found in batch output", flush=True)
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        print(f"  [generate] warning: JSON decode failed for batch ({exc})", flush=True)
        return []
    if not isinstance(data, list):
        return []

    out: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not REQUIRED_KEYS.issubset(item.keys()):
            continue
        signals = item["y_proximity_signals"]
        if isinstance(signals, str):
            signals = [signals]
        if not isinstance(signals, list):
            signals = []
        out.append(
            {
                "x": str(item["x"]).strip(),
                "y": str(item["y"]).strip(),
                "answer_x": str(item["answer_x"]).strip(),
                "answer_y": str(item["answer_y"]).strip(),
                "relation": str(item["relation"]).strip(),
                "y_proximity_signals": [str(s).strip() for s in signals],
            }
        )
    return out


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def generate_batch(
    client: LLMClient,
    fewshots: list[QuestionPair],
    domain: str,
    relation_type: tuple[str, str],
    n: int,
    seed: int,
    avoid_stems: list[str] | None = None,
    max_tokens: int = 4096,
) -> list[dict]:
    messages = build_generation_messages(fewshots, domain, relation_type, n, seed, avoid_stems)
    completion = client.complete(messages=messages, max_tokens=max_tokens, temperature=None)
    return parse_candidates(completion.content)


def generate_candidates(
    fewshots: list[QuestionPair],
    target: int,
    n_per_call: int = 4,
    seed_pool_stems: list[str] | None = None,
    client: LLMClient | None = None,
) -> list[QuestionPair]:
    """Generate up to `target` deduped candidates, rotating domain x relation-type x seed."""
    client = client or create_client(GENERATION_PROVIDER, GENERATION_MODEL)

    seen: set[str] = set(_norm(s) for s in (seed_pool_stems or []))
    accepted: list[QuestionPair] = []
    counter = 0
    seed = 0

    combos = [(d, rt) for d in DOMAINS for rt in RELATION_TYPES]
    combo_idx = 0
    empty_streak = 0

    while len(accepted) < target and empty_streak < len(combos):
        domain, rel = combos[combo_idx % len(combos)]
        combo_idx += 1
        seed += 1
        avoid = [p.x for p in accepted[-30:]]
        batch = generate_batch(client, fewshots, domain, rel, n_per_call, seed, avoid)
        added = 0
        for cand in batch:
            key = _norm(cand["x"])
            if not key or key in seen:
                continue
            if _norm(cand["answer_x"]) == _norm(cand["answer_y"]):
                continue
            seen.add(key)
            counter += 1
            pair = QuestionPair(
                id=f"gen_{counter:03d}",
                x=cand["x"],
                y=cand["y"],
                answer_x=cand["answer_x"],
                answer_y=cand["answer_y"],
                relation=cand["relation"],
                y_proximity_signals=cand["y_proximity_signals"],
            )
            accepted.append(pair)
            added += 1
            if len(accepted) >= target:
                break
        empty_streak = empty_streak + 1 if added == 0 else 0
        print(
            f"  [generate] {len(accepted)}/{target} (domain={domain.split('(')[0].strip()[:24]}, "
            f"rel={rel[0]}, +{added})",
            flush=True,
        )

    return accepted
