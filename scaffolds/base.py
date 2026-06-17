from __future__ import annotations

import json
import re
from pathlib import Path

from config import PAIRS_PATH
from models import QuestionPair, ScaffoldResult, TraceStep


def load_pairs(pair_ids: list[str] | None = None) -> list[QuestionPair]:
    raw = json.loads(PAIRS_PATH.read_text(encoding="utf-8"))
    pairs = [QuestionPair.model_validate(item) for item in raw]
    if pair_ids is None:
        return pairs
    wanted = set(pair_ids)
    return [pair for pair in pairs if pair.id in wanted]


def format_trace_for_context(trace: list[TraceStep]) -> str:
    lines: list[str] = []
    for step in trace:
        prefix = step.metadata.get("label", step.role)
        lines.append(f"[{prefix}] {step.content}")
    return "\n\n".join(lines)


def extract_final_answer(text: str) -> str:
    match = re.search(r"(?i)final answer:\s*(.+)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def save_scaffold_result(path: Path, result: ScaffoldResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_scaffold_result(path: Path) -> ScaffoldResult:
    return ScaffoldResult.model_validate_json(path.read_text(encoding="utf-8"))
