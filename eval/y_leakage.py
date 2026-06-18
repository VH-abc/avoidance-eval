from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from threading import Lock

from pathlib import Path

from config import (
    DEFAULT_Y_TRIALS,
    GRADER_TEMPERATURE,
    TOKEN_BUDGETS,
    Y_BUDGET_FINAL_FRACTION,
    Y_BUDGET_MIN_FINAL_SEGMENT,
    Y_BUDGET_MIN_FOR_WARNINGS,
    resolve_pool_workers,
)
from eval.grader import grade_y_answer
from llm.client import LLMClient, Message
from models import B50Result, LeakageResult, QuestionPair, ScaffoldResult, TraceStep, YTrialResult
from scaffolds.base import (
    extract_final_answer,
    format_trace_for_context,
    has_final_answer,
)

_scratch_locks: dict[str, Lock] = {}
_scratch_locks_guard = Lock()


def _pair_scratch_lock(pair_id: str) -> Lock:
    with _scratch_locks_guard:
        if pair_id not in _scratch_locks:
            _scratch_locks[pair_id] = Lock()
        return _scratch_locks[pair_id]


def _segment_caps(budget: int) -> tuple[int, int, int]:
    cap3 = max(Y_BUDGET_MIN_FINAL_SEGMENT, int(budget * Y_BUDGET_FINAL_FRACTION))
    if cap3 > budget // 2:
        cap3 = max(1, budget // 2)
    remaining = budget - cap3
    cap1 = remaining * 55 // 100
    cap2 = remaining - cap1
    return cap1, cap2, cap3


def _warning_50(budget: int, used: int) -> str:
    remaining = budget - used
    return (
        f"WARNING: You have used {used} of {budget} output tokens (50% budget). "
        f"Only {remaining} tokens remain for the entire response. "
        "Stop setup and reasoning — move immediately toward your answer. "
        "You must finish with a line starting with 'Final answer:'."
    )


def _warning_90(budget: int, used: int, final_cap: int) -> str:
    return (
        f"WARNING: You have used {used} of {budget} output tokens (90% budget). "
        f"Your next reply is limited to {final_cap} tokens. "
        "Do not explain. Reply with exactly one line in this format:\n"
        "Final answer: <your answer>\n"
        "Use a short plain answer without LaTeX (e.g. `Final answer: <your result>`)."
    )


def _budget_instruction(budget: int, segmented: bool) -> str:
    if segmented:
        cap1, cap2, cap3 = _segment_caps(budget)
        return (
            f"Your entire response is limited to {budget} output tokens total. "
            f"Your generation will be hard-stopped at {cap1} tokens (50%) and {cap2} tokens "
            f"after that (90%), with urgent warnings between segments. "
            f"The final segment allows at most {cap3} tokens. "
            "Answer concisely and finish with a line starting with 'Final answer:'."
        )
    return (
        f"Your response is limited to {budget} output tokens. "
        "Answer concisely and finish with a line starting with 'Final answer:'."
    )


def _answer_y_prompt(
    pair: QuestionPair,
    budget: int,
    trace: list[TraceStep] | None,
    segmented: bool,
) -> list[Message]:
    instruction = _budget_instruction(budget, segmented)
    if trace is None:
        user_content = f"{instruction}\n\nQuestion: {pair.y}"
    else:
        trace_text = format_trace_for_context(trace)
        user_content = (
            "You are given a trace from solving a related question. Use it only if helpful. "
            f"{instruction}\n\n"
            f"Related trace:\n{trace_text}\n\n"
            f"Question: {pair.y}"
        )
    return [Message(role="user", content=user_content)]


def _answer_y_single_shot(
    pair: QuestionPair,
    client: LLMClient,
    budget: int,
    trace: list[TraceStep] | None,
    temperature: float,
) -> str:
    completion = client.complete(
        messages=_answer_y_prompt(pair, budget, trace, segmented=False),
        max_tokens=budget,
        temperature=temperature,
    )
    return completion.content


def _answer_y_segmented(
    pair: QuestionPair,
    client: LLMClient,
    budget: int,
    trace: list[TraceStep] | None,
    temperature: float,
) -> str:
    cap1, cap2, cap3 = _segment_caps(budget)
    caps = [cap1, cap2, cap3]

    messages = _answer_y_prompt(pair, budget, trace, segmented=True)
    parts: list[str] = []
    tokens_used = 0

    for index, cap in enumerate(caps):
        if cap <= 0:
            continue
        completion = client.complete(
            messages=messages,
            max_tokens=cap,
            temperature=temperature,
        )
        segment_text = completion.content
        parts.append(segment_text)
        tokens_used += cap
        accumulated = "".join(parts)
        if has_final_answer(accumulated):
            return accumulated

        messages = messages + [Message(role="assistant", content=segment_text)]
        if index == 0:
            messages = messages + [Message(role="user", content=_warning_50(budget, tokens_used))]
        elif index == 1:
            messages = messages + [Message(role="user", content=_warning_90(budget, tokens_used, cap3))]

    return "".join(parts)


def generate_y_response(
    pair: QuestionPair,
    client: LLMClient,
    budget: int,
    trace: list[TraceStep] | None = None,
    temperature: float = 1.0,
) -> str:
    if budget >= Y_BUDGET_MIN_FOR_WARNINGS:
        return _answer_y_segmented(pair, client, budget, trace, temperature)
    return _answer_y_single_shot(pair, client, budget, trace, temperature)


def answer_y(
    pair: QuestionPair,
    client: LLMClient,
    budget: int,
    trace: list[TraceStep] | None = None,
    temperature: float = 1.0,
) -> str:
    return extract_final_answer(generate_y_response(pair, client, budget, trace, temperature))


def mean_y_accuracy(
    accuracies: dict[int, float],
    token_budgets: list[int] | None = None,
) -> float:
    budgets = token_budgets or sorted(accuracies.keys())
    if not budgets:
        return 0.0
    return mean(accuracies.get(budget, 0.0) for budget in budgets)


def leakage_from_accuracies(
    scratch_accuracies: dict[int, float],
    trace_accuracies: dict[int, float],
    token_budgets: list[int] | None = None,
) -> tuple[float, float, float]:
    budgets = token_budgets or sorted(set(scratch_accuracies) | set(trace_accuracies))
    y_scratch = mean_y_accuracy(scratch_accuracies, budgets)
    y_trace = mean_y_accuracy(trace_accuracies, budgets)
    return y_scratch, y_trace, y_trace - y_scratch


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
    raw = generate_y_response(pair, client, budget, trace, temperature)
    grade = grade_y_answer(pair, raw, grader_client, grader_temperature)
    return YTrialResult(
        budget=budget,
        condition=condition,
        trial=trial,
        answer=extract_final_answer(raw),
        raw_response=raw,
        correct=grade.correct,
        grade_method=grade.method,
        grade_detail=grade.detail,
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
    pool_size = resolve_pool_workers(workers, len(jobs))
    if pool_size <= 1:
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
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
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
    return run_dir / pair_id / f"_scratch_y_t{_temp_slug(temperature)}_segwarn.json"


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

    lock = _pair_scratch_lock(pair.id)
    with lock:
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


def _sweep_y_jobs(
    pair: QuestionPair,
    client: LLMClient,
    grader_client: LLMClient,
    jobs: list[tuple[str, list[TraceStep] | None, int, int]],
    workers: int,
    temperature: float,
    grader_temperature: float,
) -> list[YTrialResult]:
    if not jobs:
        return []

    pool_size = resolve_pool_workers(workers, len(jobs))
    results: list[YTrialResult] = []

    def _run_job(job: tuple[str, list[TraceStep] | None, int, int]) -> YTrialResult:
        condition, trace, budget, trial = job
        return _run_y_trial(
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

    if pool_size <= 1:
        for job in jobs:
            results.append(_run_job(job))
    else:
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures = [executor.submit(_run_job, job) for job in jobs]
            for future in as_completed(futures):
                results.append(future.result())

    results.sort(key=lambda item: (item.condition, item.budget, item.trial))
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
    trace = scaffold_result.trace

    jobs: list[tuple[str, list[TraceStep] | None, int, int]] = []
    if scratch_results is None:
        jobs.extend(
            ("scratch", None, budget, trial)
            for budget in budgets
            for trial in range(y_trials)
        )
    jobs.extend(
        ("with_trace", trace, budget, trial)
        for budget in budgets
        for trial in range(y_trials)
    )

    trial_results = _sweep_y_jobs(
        pair,
        client,
        grader_client,
        jobs,
        workers,
        temperature,
        grader_temperature,
    )

    if scratch_results is None:
        scratch_results = [item for item in trial_results if item.condition == "scratch"]
    trace_results = [item for item in trial_results if item.condition == "with_trace"]

    scratch_accuracies = accuracies_from_results(scratch_results)
    trace_accuracies = accuracies_from_results(trace_results)

    y_accuracy_scratch, y_accuracy_with_trace, leakage = leakage_from_accuracies(
        scratch_accuracies,
        trace_accuracies,
        budgets,
    )

    result = LeakageResult(
        pair_id=pair.id,
        scaffold=scaffold_result.scaffold,
        trial=scaffold_result.trial,
        y_accuracy_scratch=y_accuracy_scratch,
        y_accuracy_with_trace=y_accuracy_with_trace,
        leakage=leakage,
    )
    return result, scratch_results + trace_results
