from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean
from threading import Lock

from config import (
    DEFAULT_JOB_WORKERS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRIALS,
    DEFAULT_WORKERS,
    DEFAULT_Y_TRIALS,
    FAST_TOKEN_BUDGETS,
    GRADER_MODEL,
    GRADER_PROVIDER,
    GRADER_TEMPERATURE,
    RUNS_DIR,
    TOKEN_BUDGETS,
    Y_BUDGET_MIN_FOR_WARNINGS,
    pairs_file_for_manifest,
    resolve_pairs_path,
    resolve_pool_workers,
)
from eval.grader import grade_y_answer
from eval.x_accuracy import evaluate_x_accuracy
from eval.y_leakage import (
    evaluate_leakage,
    get_or_compute_scratch,
    leakage_from_accuracies,
)
from llm.client import LLMClient
from llm.factory import create_client
from llm.rate_limits import monitor as rate_limit_monitor
from models import LeakageResult, QuestionPair, RunSummary, ScaffoldResult, XAccuracyResult
from scaffolds.base import load_pairs, load_scaffold_result, save_pairs_for_run, save_scaffold_result
from scaffolds.baseline_avoid import run_baseline_avoid
from scaffolds.control import run_control
from scaffolds.threaded import run_threaded
from scaffolds.two_agent import run_two_agent


SCAFFOLD_RUNNERS = {
    "control": run_control,
    "baseline_avoid": run_baseline_avoid,
    "threaded": run_threaded,
    "two_agent": run_two_agent,
}


def _new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def _write_rate_limit_report(run_dir: Path, manifest: dict | None = None) -> None:
    snap = rate_limit_monitor.snapshot()
    if snap["rate_limit_hits"] == 0 and snap["slow_requests"] == 0:
        return
    (run_dir / "rate_limits.json").write_text(json.dumps(snap, indent=2), encoding="utf-8")
    if manifest is not None:
        manifest["rate_limit_hits"] = snap["rate_limit_hits"]
        manifest["slow_requests"] = snap["slow_requests"]
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _parallel_map(
    items: list,
    fn,
    job_workers: int,
    label: str,
    quiet: bool = False,
) -> list:
    if not items:
        return []

    pool_size = resolve_pool_workers(job_workers, len(items))
    if pool_size <= 1:
        return [fn(item) for item in items]

    completed = 0
    progress_lock = Lock()
    total = len(items)
    results: list = []

    def _run(item):
        nonlocal completed
        value = fn(item)
        with progress_lock:
            completed += 1
            if not quiet:
                suffix = rate_limit_monitor.progress_suffix()
                print(f"  [{label} {completed}/{total}]{suffix}", flush=True)
        return value

    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        futures = [executor.submit(_run, item) for item in items]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def _run_x_job(
    scaffold_name: str,
    pair: QuestionPair,
    trial: int,
    run_dir: Path,
    client: LLMClient,
    grader_client: LLMClient,
    temperature: float,
    grader_temperature: float,
) -> XAccuracyResult:
    result = SCAFFOLD_RUNNERS[scaffold_name](pair, client, trial, temperature)
    result_path = run_dir / pair.id / scaffold_name / f"{trial}.json"
    save_scaffold_result(result_path, result)
    return evaluate_x_accuracy(
        pair,
        scaffold_name,
        trial,
        result.answer_x,
        grader_client,
        grader_temperature,
    )


def _run_y_job(
    scaffold_name: str,
    pair: QuestionPair,
    trial: int,
    run_dir: Path,
    client: LLMClient,
    grader_client: LLMClient,
    y_trials: int,
    token_budgets: list[int] | None,
    workers: int,
    scratch_results,
    temperature: float,
    grader_temperature: float,
) -> LeakageResult:
    result_path = run_dir / pair.id / scaffold_name / f"{trial}.json"
    result = load_scaffold_result(result_path)
    leakage_result, y_trials_data = evaluate_leakage(
        pair,
        result,
        client,
        grader_client,
        y_trials=y_trials,
        token_budgets=token_budgets,
        workers=workers,
        scratch_results=scratch_results,
        temperature=temperature,
        grader_temperature=grader_temperature,
    )
    y_path = run_dir / pair.id / scaffold_name / f"{trial}_y_trials.json"
    y_path.write_text(
        json.dumps([item.model_dump() for item in y_trials_data], indent=2),
        encoding="utf-8",
    )
    return leakage_result


def run_scaffold_trial(
    scaffold_name: str,
    pair_id: str,
    trial: int,
    client: LLMClient,
    temperature: float,
    pairs_file: str | Path | None = None,
) -> ScaffoldResult:
    pairs = {pair.id: pair for pair in load_pairs(pairs_file=pairs_file)}
    pair = pairs[pair_id]
    runner = SCAFFOLD_RUNNERS[scaffold_name]
    return runner(pair, client, trial, temperature)


def run_full_eval(
    scaffolds: list[str],
    pair_ids: list[str] | None,
    trials: int,
    y_trials: int,
    provider: str,
    model: str,
    run_id: str | None = None,
    token_budgets: list[int] | None = None,
    workers: int = DEFAULT_WORKERS,
    job_workers: int = DEFAULT_JOB_WORKERS,
    skip_y: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    grader_temperature: float = GRADER_TEMPERATURE,
    pairs_file: str | Path | None = None,
) -> str:
    if run_id is None:
        run_id = _new_run_id()

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    rate_limit_monitor.reset()
    client = create_client(provider, model)
    grader_client = create_client(GRADER_PROVIDER, GRADER_MODEL)

    pairs_path = resolve_pairs_path(pairs_file)
    pairs = load_pairs(pair_ids, pairs_file=pairs_file)
    save_pairs_for_run(run_dir, pairs)
    x_results: list[XAccuracyResult] = []
    leakage_results: list[LeakageResult] = []

    manifest = {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "grader_provider": GRADER_PROVIDER,
        "grader_model": GRADER_MODEL,
        "pairs_file": pairs_file_for_manifest(pairs_path),
        "scaffolds": scaffolds,
        "pair_ids": [pair.id for pair in pairs],
        "trials": trials,
        "y_trials": y_trials,
        "token_budgets": token_budgets or TOKEN_BUDGETS,
        "workers": workers,
        "job_workers": job_workers,
        "skip_y": skip_y,
        "temperature": temperature,
        "grader_temperature": grader_temperature,
        "y_budget_warnings": True,
        "y_budget_min_for_warnings": Y_BUDGET_MIN_FOR_WARNINGS,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for scaffold_name in scaffolds:
        if scaffold_name not in SCAFFOLD_RUNNERS:
            raise ValueError(f"Unknown scaffold: {scaffold_name}")

    scaffold_jobs = [
        (scaffold_name, pair, trial)
        for scaffold_name in scaffolds
        for pair in pairs
        for trial in range(trials)
    ]

    scratch_by_pair: dict[str, list] = {}
    setup_items: list[tuple[str, object]] = []
    if not skip_y:
        for pair in pairs:
            setup_items.append(
                (
                    "scratch",
                    lambda p=pair: (
                        p.id,
                        get_or_compute_scratch(
                            p,
                            client,
                            grader_client,
                            run_dir,
                            y_trials,
                            token_budgets,
                            workers,
                            temperature,
                            grader_temperature,
                        ),
                    ),
                )
            )
    for scaffold_name, pair, trial in scaffold_jobs:
        setup_items.append(
            (
                "x",
                lambda s=scaffold_name, p=pair, t=trial: _run_x_job(
                    s,
                    p,
                    t,
                    run_dir,
                    client,
                    grader_client,
                    temperature,
                    grader_temperature,
                ),
            )
        )

    print(
        f"Running {len(setup_items)} setup tasks "
        f"(scratch baselines + X scaffolds) with unlimited job workers...",
        flush=True,
    )

    def _run_setup_item(item: tuple[str, object]):
        kind, fn = item
        return kind, fn()

    setup_results = _parallel_map(
        setup_items,
        _run_setup_item,
        job_workers,
        "setup",
    )
    for kind, value in setup_results:
        if kind == "scratch":
            pair_id, scratch = value
            scratch_by_pair[pair_id] = scratch
        else:
            x_results.append(value)

    if skip_y:
        print("Skipped Y leakage eval (--skip-y). Run y-sweep on traces when ready.", flush=True)
        _write_rate_limit_report(run_dir, manifest)
        rate_limit_monitor.print_summary()
        return run_id

    print(
        f"Running {len(scaffold_jobs)} Y leakage evals with unlimited job workers...",
        flush=True,
    )
    leakage_results = _parallel_map(
        scaffold_jobs,
        lambda job: _run_y_job(
            job[0],
            job[1],
            job[2],
            run_dir,
            client,
            grader_client,
            y_trials,
            token_budgets,
            workers,
            scratch_by_pair[job[1].id],
            temperature,
            grader_temperature,
        ),
        job_workers,
        "y-eval",
    )

    summary = build_summary(run_id, scaffolds, [pair.id for pair in pairs], x_results, leakage_results)
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    _write_rate_limit_report(run_dir, manifest)
    print_summary(summary)
    rate_limit_monitor.print_summary()
    return run_id


def build_summary(
    run_id: str,
    scaffolds: list[str],
    pair_ids: list[str],
    x_results: list[XAccuracyResult],
    leakage_results: list[LeakageResult],
) -> RunSummary:
    x_accuracy_by_scaffold: dict[str, float] = {}
    x_accuracy_by_pair: dict[str, dict[str, float]] = defaultdict(dict)

    for scaffold in scaffolds:
        scaffold_x = [r for r in x_results if r.scaffold == scaffold]
        x_accuracy_by_scaffold[scaffold] = mean(r.correct for r in scaffold_x) if scaffold_x else 0.0
        for pair_id in pair_ids:
            pair_x = [r for r in scaffold_x if r.pair_id == pair_id]
            if pair_x:
                x_accuracy_by_pair[pair_id][scaffold] = mean(r.correct for r in pair_x)

    mean_y_accuracy_scratch: dict[str, float | None] = {}
    mean_y_accuracy_with_trace: dict[str, float | None] = {}
    mean_leakage: dict[str, float | None] = {}
    leakage_by_pair: dict[str, dict[str, float | None]] = defaultdict(dict)

    for scaffold in scaffolds:
        scaffold_leakage = [r for r in leakage_results if r.scaffold == scaffold]
        mean_y_accuracy_scratch[scaffold] = _mean_or_none(
            [r.y_accuracy_scratch for r in scaffold_leakage]
        )
        mean_y_accuracy_with_trace[scaffold] = _mean_or_none(
            [r.y_accuracy_with_trace for r in scaffold_leakage]
        )
        mean_leakage[scaffold] = _mean_or_none([r.leakage for r in scaffold_leakage])
        for pair_id in pair_ids:
            pair_leakage = [r for r in scaffold_leakage if r.pair_id == pair_id]
            if pair_leakage:
                leakage_by_pair[pair_id][scaffold] = mean(r.leakage for r in pair_leakage)

    return RunSummary(
        run_id=run_id,
        scaffolds=scaffolds,
        pair_ids=pair_ids,
        x_accuracy_by_scaffold=x_accuracy_by_scaffold,
        x_accuracy_by_pair=dict(x_accuracy_by_pair),
        mean_y_accuracy_scratch=mean_y_accuracy_scratch,
        mean_y_accuracy_with_trace=mean_y_accuracy_with_trace,
        mean_leakage=mean_leakage,
        leakage_by_pair=dict(leakage_by_pair),
    )


def analyze_run(
    run_id: str,
    job_workers: int = DEFAULT_JOB_WORKERS,
) -> RunSummary:
    run_dir = RUNS_DIR / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    rate_limit_monitor.reset()
    grader_client = create_client(manifest["grader_provider"], manifest["grader_model"])
    pairs = {
        pair.id: pair
        for pair in load_pairs(
            manifest["pair_ids"],
            pairs_file=manifest.get("pairs_file"),
        )
    }
    grader_temperature = manifest.get("grader_temperature", GRADER_TEMPERATURE)
    token_budgets = manifest.get("token_budgets", TOKEN_BUDGETS)
    job_workers = manifest.get("job_workers", job_workers)
    y_workers = manifest.get("workers", DEFAULT_WORKERS)

    tasks: list[tuple[Path, str, str]] = []
    for pair_id in manifest["pair_ids"]:
        for scaffold_name in manifest["scaffolds"]:
            scaffold_dir = run_dir / pair_id / scaffold_name
            if not scaffold_dir.exists():
                continue
            for result_path in sorted(scaffold_dir.glob("*.json")):
                if result_path.name.endswith("_y_trials.json"):
                    continue
                tasks.append((result_path, pair_id, scaffold_name))

    def _analyze_task(task: tuple[Path, str, str]) -> tuple[XAccuracyResult, LeakageResult | None]:
        result_path, pair_id, scaffold_name = task
        result = load_scaffold_result(result_path)
        pair = pairs[pair_id]
        x_result = evaluate_x_accuracy(
            pair,
            scaffold_name,
            result.trial,
            result.answer_x,
            grader_client,
            grader_temperature,
        )
        y_path = result_path.with_name(f"{result.trial}_y_trials.json")
        if not y_path.exists():
            return x_result, None

        y_data = json.loads(y_path.read_text(encoding="utf-8"))

        def _regrade_item(item: dict) -> dict:
            grade = grade_y_answer(
                pair,
                item["answer"],
                grader_client,
                grader_temperature,
            )
            item["correct"] = grade.correct
            item["grade_method"] = grade.method
            item["grade_detail"] = grade.detail
            return item

        y_data = _parallel_map(y_data, _regrade_item, y_workers, "y-grade", quiet=True)
        scratch_acc: dict[int, list[bool]] = defaultdict(list)
        trace_acc: dict[int, list[bool]] = defaultdict(list)
        for item in y_data:
            if item["condition"] == "scratch":
                scratch_acc[item["budget"]].append(item["correct"])
            else:
                trace_acc[item["budget"]].append(item["correct"])
        y_path.write_text(json.dumps(y_data, indent=2), encoding="utf-8")

        scratch_accuracies = {
            budget: mean(vals) for budget, vals in scratch_acc.items()
        }
        trace_accuracies = {
            budget: mean(vals) for budget, vals in trace_acc.items()
        }
        y_accuracy_scratch, y_accuracy_with_trace, leakage = leakage_from_accuracies(
            scratch_accuracies,
            trace_accuracies,
            token_budgets,
        )
        return x_result, LeakageResult(
            pair_id=pair_id,
            scaffold=scaffold_name,
            trial=result.trial,
            y_accuracy_scratch=y_accuracy_scratch,
            y_accuracy_with_trace=y_accuracy_with_trace,
            leakage=leakage,
        )

    print(f"Regrading {len(tasks)} traces...", flush=True)
    analyzed = _parallel_map(tasks, _analyze_task, job_workers, "analyze")
    x_results = [item[0] for item in analyzed]
    leakage_results = [item[1] for item in analyzed if item[1] is not None]

    summary = build_summary(
        run_id,
        manifest["scaffolds"],
        manifest["pair_ids"],
        x_results,
        leakage_results,
    )
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    _write_rate_limit_report(run_dir, manifest)
    print_summary(summary)
    rate_limit_monitor.print_summary()
    return summary


def y_sweep_trace(
    trace_path: Path,
    provider: str,
    model: str,
    y_trials: int,
    token_budgets: list[int] | None = None,
    workers: int = DEFAULT_WORKERS,
    run_dir: Path | None = None,
    temperature: float = DEFAULT_TEMPERATURE,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> LeakageResult:
    result = load_scaffold_result(trace_path)
    pairs_file = None
    if run_dir is not None:
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            pairs_file = manifest.get("pairs_file")
    pair = {pair.id: pair for pair in load_pairs(pairs_file=pairs_file)}[result.pair_id]
    rate_limit_monitor.reset()
    client = create_client(provider, model)
    grader_client = create_client(GRADER_PROVIDER, GRADER_MODEL)

    scratch_results = None
    if run_dir is not None:
        from eval.y_leakage import get_or_compute_scratch

        scratch_results = get_or_compute_scratch(
            pair,
            client,
            grader_client,
            run_dir,
            y_trials,
            token_budgets,
            workers,
            temperature,
            grader_temperature,
        )

    leakage_result, y_data = evaluate_leakage(
        pair,
        result,
        client,
        grader_client,
        y_trials=y_trials,
        token_budgets=token_budgets,
        workers=workers,
        scratch_results=scratch_results,
        temperature=temperature,
        grader_temperature=grader_temperature,
    )
    out_path = trace_path.with_name(f"{result.trial}_y_trials.json")
    out_path.write_text(
        json.dumps([item.model_dump() for item in y_data], indent=2),
        encoding="utf-8",
    )
    if run_dir is not None:
        _write_rate_limit_report(run_dir)
    print(json.dumps(leakage_result.model_dump(), indent=2))
    rate_limit_monitor.print_summary()
    return leakage_result


def print_summary(summary: RunSummary) -> None:
    headers = ["Scaffold", "X accuracy", "Y acc scratch", "Y acc w/ trace", "Leakage"]
    rows: list[list[str]] = []
    for scaffold in summary.scaffolds:
        rows.append(
            [
                scaffold,
                f"{summary.x_accuracy_by_scaffold.get(scaffold, 0.0):.2f}",
                _fmt(summary.mean_y_accuracy_scratch.get(scaffold)),
                _fmt(summary.mean_y_accuracy_with_trace.get(scaffold)),
                _fmt(summary.mean_leakage.get(scaffold)),
            ]
        )

    widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    line = " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    print(line)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"
