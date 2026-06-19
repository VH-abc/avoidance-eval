from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from config import FAST_TOKEN_BUDGETS, GRADER_TEMPERATURE
from eval.grader import grade_x_answer
from eval.y_leakage import mean_y_accuracy, sweep_y_condition
from llm.client import LLMClient
from models import QuestionPair, TraceStep
from scaffolds.control import run_control


@dataclass
class PairMetrics:
    pair_id: str
    y_acc_scratch: float
    y_acc_full_trace: float
    y_acc_answer_only: float
    trace_leakage: float
    answer_only_leakage: float
    x_scratch_acc: float
    n_x_trials: int
    used_correct_trace: bool
    acc_by_budget: dict[str, dict[int, float]] = field(default_factory=dict)
    x_answers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "y_acc_scratch": self.y_acc_scratch,
            "y_acc_full_trace": self.y_acc_full_trace,
            "y_acc_answer_only": self.y_acc_answer_only,
            "trace_leakage": self.trace_leakage,
            "answer_only_leakage": self.answer_only_leakage,
            "x_scratch_acc": self.x_scratch_acc,
            "n_x_trials": self.n_x_trials,
            "used_correct_trace": self.used_correct_trace,
            "acc_by_budget": {
                cond: {str(b): a for b, a in by_b.items()}
                for cond, by_b in self.acc_by_budget.items()
            },
            "x_answers": self.x_answers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PairMetrics":
        return cls(
            pair_id=data["pair_id"],
            y_acc_scratch=data["y_acc_scratch"],
            y_acc_full_trace=data["y_acc_full_trace"],
            y_acc_answer_only=data["y_acc_answer_only"],
            trace_leakage=data["trace_leakage"],
            answer_only_leakage=data["answer_only_leakage"],
            x_scratch_acc=data["x_scratch_acc"],
            n_x_trials=data.get("n_x_trials", 0),
            used_correct_trace=data.get("used_correct_trace", False),
            acc_by_budget={
                cond: {int(b): a for b, a in by_b.items()}
                for cond, by_b in data.get("acc_by_budget", {}).items()
            },
            x_answers=data.get("x_answers", []),
        )


def _answer_only_trace(pair: QuestionPair) -> list[TraceStep]:
    return [
        TraceStep(
            role="assistant",
            content=f"Final answer: {pair.answer_x}",
            metadata={"label": "x_answer_only", "answer_only": True},
        )
    ]


def measure_pair(
    pair: QuestionPair,
    solver_client: LLMClient,
    grader_client: LLMClient,
    y_trials: int = 4,
    x_trials: int = 3,
    token_budgets: list[int] | None = None,
    workers: int = 0,
    temperature: float = 1.0,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> PairMetrics:
    """Measure scratch / full-trace / answer-only Y accuracy for one pair.

    - The full-trace condition uses an actual solver solution of X (preferring a
      trace that solves X correctly), modeling "the easiest approach's trace".
    - The answer-only condition shows only the gold answer to X, modeling
      "Y after seeing the answer to X".
    """
    budgets = token_budgets or FAST_TOKEN_BUDGETS

    # Solve X several times: measure X hardness and pick a representative trace.
    x_results = [run_control(pair, solver_client, trial, temperature) for trial in range(x_trials)]
    x_grades = [grade_x_answer(pair, r.answer_x, grader_client, grader_temperature) for r in x_results]
    x_scratch_acc = sum(g.correct for g in x_grades) / len(x_grades) if x_grades else 0.0

    correct_idx = next((i for i, g in enumerate(x_grades) if g.correct), None)
    used_correct_trace = correct_idx is not None
    trace_source = x_results[correct_idx] if correct_idx is not None else x_results[0]
    full_trace = trace_source.trace
    answer_only = _answer_only_trace(pair)

    scratch_acc, _ = sweep_y_condition(
        pair, solver_client, grader_client, "scratch", None,
        y_trials, budgets, workers, temperature, grader_temperature,
    )
    full_acc, _ = sweep_y_condition(
        pair, solver_client, grader_client, "with_trace", full_trace,
        y_trials, budgets, workers, temperature, grader_temperature,
    )
    answer_acc, _ = sweep_y_condition(
        pair, solver_client, grader_client, "answer_only", answer_only,
        y_trials, budgets, workers, temperature, grader_temperature,
    )

    y_scratch = mean_y_accuracy(scratch_acc, budgets)
    y_full = mean_y_accuracy(full_acc, budgets)
    y_answer = mean_y_accuracy(answer_acc, budgets)

    return PairMetrics(
        pair_id=pair.id,
        y_acc_scratch=y_scratch,
        y_acc_full_trace=y_full,
        y_acc_answer_only=y_answer,
        trace_leakage=y_full - y_scratch,
        answer_only_leakage=y_answer - y_scratch,
        x_scratch_acc=x_scratch_acc,
        n_x_trials=x_trials,
        used_correct_trace=used_correct_trace,
        acc_by_budget={"scratch": scratch_acc, "with_trace": full_acc, "answer_only": answer_acc},
        x_answers=[r.answer_x for r in x_results],
    )


def measure_pairs(
    pairs: list[QuestionPair],
    solver_client: LLMClient,
    grader_client: LLMClient,
    y_trials: int = 4,
    x_trials: int = 3,
    token_budgets: list[int] | None = None,
    sweep_workers: int = 4,
    pair_workers: int = 8,
    temperature: float = 1.0,
    grader_temperature: float = GRADER_TEMPERATURE,
    progress_label: str = "measure",
) -> list[PairMetrics]:
    """Measure many pairs with pair-level parallelism.

    Total concurrency is roughly pair_workers * sweep_workers; keep the product
    modest to avoid rate limits.
    """
    if not pairs:
        return []

    def _one(pair: QuestionPair) -> PairMetrics:
        return measure_pair(
            pair, solver_client, grader_client,
            y_trials=y_trials, x_trials=x_trials, token_budgets=token_budgets,
            workers=sweep_workers, temperature=temperature, grader_temperature=grader_temperature,
        )

    results: list[PairMetrics] = []
    pool = max(1, min(pair_workers, len(pairs)))
    completed = 0
    total = len(pairs)
    if pool <= 1:
        for pair in pairs:
            results.append(_one(pair))
            completed += 1
            print(f"  [{progress_label} {completed}/{total}] {pair.id}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=pool) as executor:
            futures = {executor.submit(_one, pair): pair for pair in pairs}
            for future in as_completed(futures):
                metrics = future.result()
                results.append(metrics)
                completed += 1
                print(f"  [{progress_label} {completed}/{total}] {metrics.pair_id}", flush=True)
    order = {pair.id: i for i, pair in enumerate(pairs)}
    results.sort(key=lambda m: order.get(m.pair_id, 0))
    return results
