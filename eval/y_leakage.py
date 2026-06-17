from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean

from pathlib import Path

from config import DEFAULT_Y_TRIALS, GRADER_TEMPERATURE, TOKEN_BUDGETS
from eval.grader import grade_y_answer
from llm.client import LLMClient, Message
from models import B50Result, LeakageResult, QuestionPair, ScaffoldResult, TraceStep, YTrialResult
from scaffolds.base import extract_final_answer, format_trace_for_context


def _budget_instruction(budget: int) -> str:
    return (
        f"Your response is limited to {budget} output tokens. "
        "You have been warned of this limit — answer comfortably within it, "
        "staying concise enough to finish with a line starting with 'Final answer:'."
    )


def _answer_y_prompt(
    pair: QuestionPair,
    budget: int,
    trace: list[TraceStep] | None,
) -> list[Message]:
    if trace is None:
        user_content = (
            f"{_budget_instruction(budget)}\n\n"
            f"Question: {pair.y}"
        )
    else:
        trace_text = format_trace_for_context(trace)
        user_content = (
            "You are given a trace from solving a related question. Use it only if helpful. "
            f"{_budget_instruction(budget)}\n\n"
            f"Related trace:\n{trace_text}\n\n"
            f"Question: {pair.y}"
        )
    return [Message(role="user", content=user_content)]


def answer_y(
    pair: QuestionPair,
    client: LLMClient,
    budget: int,
    trace: list[TraceStep] | None = None,
    temperature: float = 1.0,
) -> str:
    completion = client.complete(
        messages=_answer_y_prompt(pair, budget, trace),
        max_tokens=budget,
        temperature=temperature,
    )
    return extract_final_answer(completion.content)


def compute_b50(
    accuracies: dict[int, float],
    token_budgets: list[int] | None = None,
) -> B50Result:
    budgets = sorted(token_budgets or TOKEN_BUDGETS)
    budget_accs = {budget: accuracies.get(budget, 0.0) for budget in budgets}

    for budget in budgets:
        if budget_accs[budget] >= 0.5:
            prev_budgets = [b for b in budgets if b < budget]
            if not prev_budgets:
                return B50Result(b50=float(budget), accuracies=budget_accs)
            prev_budget = prev_budgets[-1]
            prev_acc = budget_accs[prev_budget]
            curr_acc = budget_accs[budget]
            if curr_acc == prev_acc:
                return B50Result(b50=float(budget), accuracies=budget_accs)
            fraction = (0.5 - prev_acc) / (curr_acc - prev_acc)
            interpolated = prev_budget + fraction * (budget - prev_budget)
            return B50Result(b50=interpolated, accuracies=budget_accs)

    if budget_accs[budgets[-1]] > 0:
        return B50Result(
            b50=None,
            accuracies=budget_accs,
            undefined_reason="accuracy never reached 50%",
        )
    return B50Result(
        b50=None,
        accuracies=budget_accs,
        undefined_reason="zero accuracy at all budgets",
    )


def _run_y_trial(
    pair: QuestionPair,
    client: LLMClient,
    grader_client: LLMClient,
    condition: str,
    budget: int,
    trial: int,
    trace: list[TraceStep] | None,
    temperature: float,
    grader_temperature: float,
) -> YTrialResult:
    answer = answer_y(pair, client, budget, trace, temperature)
    grade = grade_y_answer(pair, answer, grader_client, grader_temperature)
    return YTrialResult(
        budget=budget,
        condition=condition,
        trial=trial,
        answer=answer,
        correct=grade.correct,
    )


def sweep_y_condition(
    pair: QuestionPair,
    client: LLMClient,
    grader_client: LLMClient,
    condition: str,
    trace: list[TraceStep] | None,
    y_trials: int,
    token_budgets: list[int] | None = None,
    workers: int = 1,
    temperature: float = 1.0,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> tuple[dict[int, float], list[YTrialResult]]:
    budgets = token_budgets or TOKEN_BUDGETS
    jobs = [
        (budget, trial)
        for budget in budgets
        for trial in range(y_trials)
    ]

    results: list[YTrialResult] = []
    if workers <= 1:
        for budget, trial in jobs:
            results.append(
                _run_y_trial(
                    pair,
                    client,
                    grader_client,
                    condition,
                    budget,
                    trial,
                    trace,
                    temperature,
                    grader_temperature,
                )
            )
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_y_trial,
                    pair,
                    client,
                    grader_client,
                    condition,
                    budget,
                    trial,
                    trace,
                    temperature,
                    grader_temperature,
                ): (budget, trial)
                for budget, trial in jobs
            }
            for future in as_completed(futures):
                results.append(future.result())

    by_budget: dict[int, list[bool]] = defaultdict(list)
    for result in results:
        by_budget[result.budget].append(result.correct)

    accuracies = {
        budget: mean(values) if values else 0.0
        for budget, values in by_budget.items()
    }
    results.sort(key=lambda item: (item.budget, item.trial))
    return accuracies, results


def _temp_slug(temperature: float) -> str:
    return str(temperature).replace(".", "p")


def scratch_cache_path(run_dir: Path, pair_id: str, temperature: float) -> Path:
    return run_dir / pair_id / f"_scratch_y_t{_temp_slug(temperature)}.json"


def load_scratch_results(path) -> list[YTrialResult]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [YTrialResult.model_validate(item) for item in raw]


def save_scratch_results(path, results: list[YTrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.model_dump() for item in results], indent=2),
        encoding="utf-8",
    )


def accuracies_from_results(results: list[YTrialResult]) -> dict[int, float]:
    by_budget: dict[int, list[bool]] = defaultdict(list)
    for result in results:
        by_budget[result.budget].append(result.correct)
    return {
        budget: mean(values) if values else 0.0
        for budget, values in by_budget.items()
    }


def get_or_compute_scratch(
    pair: QuestionPair,
    client: LLMClient,
    grader_client: LLMClient,
    run_dir,
    y_trials: int,
    token_budgets: list[int] | None,
    workers: int,
    temperature: float = 1.0,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> list[YTrialResult]:
    path = scratch_cache_path(run_dir, pair.id, temperature)
    if path.exists():
        return load_scratch_results(path)

    _, results = sweep_y_condition(
        pair,
        client,
        grader_client,
        "scratch",
        None,
        y_trials,
        token_budgets=token_budgets,
        workers=workers,
        temperature=temperature,
        grader_temperature=grader_temperature,
    )
    save_scratch_results(path, results)
    return results


def evaluate_leakage(
    pair: QuestionPair,
    scaffold_result: ScaffoldResult,
    client: LLMClient,
    grader_client: LLMClient,
    y_trials: int = DEFAULT_Y_TRIALS,
    token_budgets: list[int] | None = None,
    workers: int = 1,
    scratch_results: list[YTrialResult] | None = None,
    temperature: float = 1.0,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> tuple[LeakageResult, list[YTrialResult]]:
    budgets = token_budgets or TOKEN_BUDGETS

    if scratch_results is None:
        scratch_accuracies, scratch_results = sweep_y_condition(
            pair,
            client,
            grader_client,
            "scratch",
            None,
            y_trials,
            token_budgets=budgets,
            workers=workers,
            temperature=temperature,
            grader_temperature=grader_temperature,
        )
    else:
        scratch_accuracies = accuracies_from_results(scratch_results)

    trace_accuracies, trace_results = sweep_y_condition(
        pair,
        client,
        grader_client,
        "with_trace",
        scaffold_result.trace,
        y_trials,
        token_budgets=budgets,
        workers=workers,
        temperature=temperature,
        grader_temperature=grader_temperature,
    )

    b50_scratch = compute_b50(scratch_accuracies, budgets)
    b50_trace = compute_b50(trace_accuracies, budgets)

    leakage: float | None = None
    undefined_reason: str | None = None
    if b50_scratch.b50 is None:
        undefined_reason = b50_scratch.undefined_reason
    elif b50_trace.b50 is None:
        undefined_reason = "B50 with trace undefined"
    else:
        leakage = (b50_scratch.b50 - b50_trace.b50) / b50_scratch.b50

    result = LeakageResult(
        pair_id=pair.id,
        scaffold=scaffold_result.scaffold,
        trial=scaffold_result.trial,
        b50_scratch=b50_scratch.b50,
        b50_with_trace=b50_trace.b50,
        leakage=leakage,
        undefined_reason=undefined_reason,
    )
    return result, scratch_results + trace_results
