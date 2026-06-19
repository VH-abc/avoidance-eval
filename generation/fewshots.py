from __future__ import annotations

import json
import re
from pathlib import Path

from config import DATA_DIR, FAST_TOKEN_BUDGETS, GRADER_MODEL, GRADER_PROVIDER, SOLVER_MODEL, SOLVER_PROVIDER
from generation.measure import PairMetrics, measure_pairs
from generation.scoring import FEWSHOT_THRESHOLDS, FilterThresholds, passes, score
from llm.factory import create_client
from models import QuestionPair
from scaffolds.base import load_pairs

DEFAULT_POOL_FILES = [
    DATA_DIR / "pairs_structural_v3.json",
    DATA_DIR / "pairs.json",
    DATA_DIR / "pairs_shared_structure.json",
]
FEWSHOT_METRICS_PATH = DATA_DIR / "_fewshot_metrics.json"
FEWSHOT_SELECTED_PATH = DATA_DIR / "_fewshots_selected.json"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.lower())).strip()


def load_candidate_pool(paths: list[Path] | None = None) -> list[QuestionPair]:
    paths = paths or DEFAULT_POOL_FILES
    seen_keys: set[tuple[str, str]] = set()
    seen_ids: set[str] = set()
    pool: list[QuestionPair] = []
    for path in paths:
        if not path.exists():
            continue
        for pair in load_pairs(pairs_file=path):
            key = (_norm(pair.x), _norm(pair.y))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            pair_id = pair.id
            while pair_id in seen_ids:
                pair_id = f"{pair_id}_dup"
            seen_ids.add(pair_id)
            if pair_id != pair.id:
                pair = pair.model_copy(update={"id": pair_id})
            pool.append(pair)
    return pool


def select_fewshots(
    metrics: list[PairMetrics],
    thresholds: FilterThresholds = FEWSHOT_THRESHOLDS,
    max_n: int = 8,
) -> list[PairMetrics]:
    eligible = [m for m in metrics if passes(m, thresholds)]
    eligible.sort(key=score, reverse=True)
    return eligible[:max_n]


def run_stage0(
    y_trials: int = 4,
    x_trials: int = 3,
    token_budgets: list[int] | None = None,
    max_fewshots: int = 8,
) -> tuple[list[PairMetrics], list[QuestionPair]]:
    budgets = token_budgets or FAST_TOKEN_BUDGETS
    pool = load_candidate_pool()
    print(f"Stage 0: measuring {len(pool)} candidate few-shot pairs ...", flush=True)
    solver = create_client(SOLVER_PROVIDER, SOLVER_MODEL)
    grader = create_client(GRADER_PROVIDER, GRADER_MODEL)
    metrics = measure_pairs(
        pool, solver, grader,
        y_trials=y_trials, x_trials=x_trials, token_budgets=budgets,
        progress_label="fewshot-measure",
    )

    by_id = {p.id: p for p in pool}
    ranked = sorted(metrics, key=score, reverse=True)
    print("\nRanked few-shot candidates (trace_leak / ans_leak / y_scratch / x_acc / score):")
    for m in ranked:
        tag = "KEEP" if passes(m, FEWSHOT_THRESHOLDS) else "drop"
        print(
            f"  [{tag}] {m.pair_id:<38} "
            f"{m.trace_leakage:+.2f} / {m.answer_only_leakage:+.2f} / "
            f"{m.y_acc_scratch:.2f} / {m.x_scratch_acc:.2f} / {score(m):+.2f}"
        )

    selected = select_fewshots(metrics, FEWSHOT_THRESHOLDS, max_fewshots)
    selected_pairs = [by_id[m.pair_id] for m in selected]

    FEWSHOT_METRICS_PATH.write_text(
        json.dumps([m.to_dict() for m in ranked], indent=2), encoding="utf-8"
    )
    FEWSHOT_SELECTED_PATH.write_text(
        json.dumps(
            [
                {**by_id[m.pair_id].model_dump(), "_metrics": m.to_dict()}
                for m in selected
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSelected {len(selected_pairs)} few-shots -> {FEWSHOT_SELECTED_PATH}")
    return metrics, selected_pairs


def load_selected_fewshots() -> list[QuestionPair]:
    if not FEWSHOT_SELECTED_PATH.exists():
        return []
    raw = json.loads(FEWSHOT_SELECTED_PATH.read_text(encoding="utf-8"))
    pairs: list[QuestionPair] = []
    for item in raw:
        data = {k: v for k, v in item.items() if not k.startswith("_")}
        pairs.append(QuestionPair.model_validate(data))
    return pairs


if __name__ == "__main__":
    from generation.spend import configure_spend, current_spend
    from config import GENERATION_MAX_SPEND

    configure_spend(GENERATION_MAX_SPEND)
    run_stage0()
    print(f"Stage 0 spend: ${current_spend():.4f}")
