from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from config import GENERATION_MODEL, GENERATION_PROVIDER
from eval.grader import answers_equivalent
from generation.prompts import build_verification_messages
from llm.client import LLMClient
from llm.factory import create_client
from models import QuestionPair


@dataclass
class VerifyRecord:
    pair_id: str
    accepted: bool
    reason: str
    proposed_answer_x: str
    proposed_answer_y: str
    sample_answers: list[dict] = field(default_factory=list)


def _iter_balanced_objects(text: str) -> list[str]:
    out: list[str] = []
    depth = 0
    in_str = False
    escape = False
    start = -1
    for i, ch in enumerate(text):
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
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    out.append(text[start : i + 1])
    return out


def extract_last_json_object(text: str) -> dict | None:
    for blob in reversed(_iter_balanced_objects(text)):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "answer_x" in data and "answer_y" in data:
            return data
    return None


def _one_sample(client: LLMClient, pair: QuestionPair, max_tokens: int) -> dict | None:
    messages = build_verification_messages(pair.x, pair.y)
    completion = client.complete(messages=messages, max_tokens=max_tokens, temperature=None)
    return extract_last_json_object(completion.content)


def verify_pair(
    client: LLMClient,
    pair: QuestionPair,
    samples: int = 2,
    max_tokens: int = 1200,
) -> VerifyRecord:
    parsed: list[dict] = []
    for _ in range(samples):
        obj = _one_sample(client, pair, max_tokens)
        if obj is not None:
            parsed.append(obj)

    if len(parsed) < samples:
        return VerifyRecord(
            pair.id, False, "verification output unparseable",
            pair.answer_x, pair.answer_y, parsed,
        )

    well_posed = all(
        bool(obj.get("x_is_well_posed", True)) and bool(obj.get("y_is_well_posed", True))
        for obj in parsed
    )
    if not well_posed:
        return VerifyRecord(
            pair.id, False, "model flagged a question as not well-posed",
            pair.answer_x, pair.answer_y, parsed,
        )

    x_ok = all(answers_equivalent(str(obj.get("answer_x", "")), pair.answer_x) for obj in parsed)
    y_ok = all(answers_equivalent(str(obj.get("answer_y", "")), pair.answer_y) for obj in parsed)

    if x_ok and y_ok:
        return VerifyRecord(pair.id, True, "verified", pair.answer_x, pair.answer_y, parsed)

    mismatch = []
    if not x_ok:
        mismatch.append("answer_x")
    if not y_ok:
        mismatch.append("answer_y")
    return VerifyRecord(
        pair.id, False, f"verification disagreed on {', '.join(mismatch)}",
        pair.answer_x, pair.answer_y, parsed,
    )


def verify_pairs(
    pairs: list[QuestionPair],
    client: LLMClient | None = None,
    samples: int = 2,
    max_tokens: int = 1200,
    workers: int = 6,
) -> tuple[list[QuestionPair], list[VerifyRecord]]:
    client = client or create_client(GENERATION_PROVIDER, GENERATION_MODEL)
    records: list[VerifyRecord] = []
    by_id = {p.id: p for p in pairs}

    pool = max(1, min(workers, len(pairs))) if pairs else 1
    completed = 0
    total = len(pairs)
    if pool <= 1:
        for pair in pairs:
            records.append(verify_pair(client, pair, samples, max_tokens))
            completed += 1
            print(f"  [verify {completed}/{total}] {pair.id} -> {records[-1].accepted}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=pool) as executor:
            futures = {executor.submit(verify_pair, client, p, samples, max_tokens): p for p in pairs}
            for future in as_completed(futures):
                rec = future.result()
                records.append(rec)
                completed += 1
                print(f"  [verify {completed}/{total}] {rec.pair_id} -> {rec.accepted}", flush=True)

    accepted = [by_id[r.pair_id] for r in records if r.accepted]
    return accepted, records
