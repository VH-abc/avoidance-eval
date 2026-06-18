from __future__ import annotations

import json
import re
from pathlib import Path

from config import DATA_DIR, resolve_pairs_path
from models import QuestionPair, ScaffoldResult, TraceStep


def load_pairs(
    pair_ids: list[str] | None = None,
    pairs_file: str | Path | None = None,
) -> list[QuestionPair]:
    raw = json.loads(resolve_pairs_path(pairs_file).read_text(encoding="utf-8"))
    pairs = [QuestionPair.model_validate(item) for item in raw]
    if pair_ids is None:
        return pairs
    wanted = set(pair_ids)
    return [pair for pair in pairs if pair.id in wanted]


def save_pairs_for_run(run_dir: Path, pairs: list[QuestionPair]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "pairs.json").write_text(
        json.dumps([pair.model_dump() for pair in pairs], indent=2),
        encoding="utf-8",
    )


def load_pairs_for_run(run_id: str) -> list[QuestionPair]:
    from config import RUNS_DIR

    run_dir = RUNS_DIR / run_id
    run_pairs_path = run_dir / "pairs.json"
    if run_pairs_path.exists():
        raw = json.loads(run_pairs_path.read_text(encoding="utf-8"))
        return [QuestionPair.model_validate(item) for item in raw]

    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return load_pairs()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pairs = load_pairs(manifest.get("pair_ids"), pairs_file=manifest.get("pairs_file"))
    if pairs:
        save_pairs_for_run(run_dir, pairs)
    return pairs


def load_pair_for_run(run_id: str, pair_id: str) -> QuestionPair | None:
    for pair in load_pairs_for_run(run_id):
        if pair.id == pair_id:
            return pair
    for path in sorted(DATA_DIR.glob("*.json")):
        matches = load_pairs([pair_id], pairs_file=path)
        if matches:
            return matches[0]
    return None


def format_trace_for_context(trace: list[TraceStep]) -> str:
    lines: list[str] = []
    for step in trace:
        prefix = step.metadata.get("label", step.role)
        lines.append(f"[{prefix}] {step.content}")
    return "\n\n".join(lines)


_FINAL_ANSWER_RE = re.compile(r"(?i)final answer:\s*(.+)", re.DOTALL)


def has_final_answer(text: str) -> bool:
    return _FINAL_ANSWER_RE.search(text) is not None


def extract_final_answer(text: str) -> str:
    match = _FINAL_ANSWER_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def save_scaffold_result(path: Path, result: ScaffoldResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")


def load_scaffold_result(path: Path) -> ScaffoldResult:
    return ScaffoldResult.model_validate_json(path.read_text(encoding="utf-8"))
