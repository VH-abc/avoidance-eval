from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from config import (
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
)
from eval.x_accuracy import evaluate_x_accuracy
from eval.y_leakage import (
    compute_b50,
    evaluate_leakage,
    get_or_compute_scratch,
)
from llm.client import LLMClient
from llm.factory import create_client
from models import LeakageResult, RunSummary, ScaffoldResult, XAccuracyResult
from scaffolds.base import load_pairs, load_scaffold_result, save_scaffold_result
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


def run_scaffold_trial(
    scaffold_name: str,
    pair_id: str,
    trial: int,
    client: LLMClient,
    temperature: float,
) -> ScaffoldResult:
    pairs = {pair.id: pair for pair in load_pairs()}
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
    skip_y: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> str:
    if run_id is None:
        run_id = _new_run_id()

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    client = create_client(provider, model)
    grader_client = create_client(GRADER_PROVIDER, GRADER_MODEL)

    pairs = load_pairs(pair_ids)
    x_results: list[XAccuracyResult] = []
    leakage_results: list[LeakageResult] = []

    manifest = {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "grader_provider": GRADER_PROVIDER,
        "grader_model": GRADER_MODEL,
        "scaffolds": scaffolds,
        "pair_ids": [pair.id for pair in pairs],
        "trials": trials,
        "y_trials": y_trials,
        "token_budgets": token_budgets or TOKEN_BUDGETS,
        "workers": workers,
        "skip_y": skip_y,
        "temperature": temperature,
        "grader_temperature": grader_temperature,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    scratch_by_pair: dict[str, list] = {}
    if not skip_y:
        print("Precomputing scratch Y baselines (once per pair)...", flush=True)
        for pair in pairs:
            scratch_by_pair[pair.id] = get_or_compute_scratch(
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
            print(f"  scratch baseline ready: {pair.id}", flush=True)

    total_jobs = len(scaffolds) * len(pairs) * trials
    job_num = 0

    for scaffold_name in scaffolds:
        if scaffold_name not in SCAFFOLD_RUNNERS:
            raise ValueError(f"Unknown scaffold: {scaffold_name}")

        for pair in pairs:
            for trial in range(trials):
                job_num += 1
                print(
                    f"[{job_num}/{total_jobs}] {pair.id} / {scaffold_name} / trial {trial}",
                    flush=True,
                )
                result = SCAFFOLD_RUNNERS[scaffold_name](pair, client, trial, temperature)
                result_path = run_dir / pair.id / scaffold_name / f"{trial}.json"
                save_scaffold_result(result_path, result)

                x_result = evaluate_x_accuracy(
                    pair,
                    scaffold_name,
                    trial,
                    result.answer_x,
                    grader_client,
                    grader_temperature,
                )
                x_results.append(x_result)

                if skip_y:
                    continue

                leakage_result, y_trials_data = evaluate_leakage(
                    pair,
                    result,
                    client,
                    grader_client,
                    y_trials=y_trials,
                    token_budgets=token_budgets,
                    workers=workers,
                    scratch_results=scratch_by_pair[pair.id],
                    temperature=temperature,
                    grader_temperature=grader_temperature,
                )
                leakage_results.append(leakage_result)

                y_path = run_dir / pair.id / scaffold_name / f"{trial}_y_trials.json"
                y_path.write_text(
                    json.dumps([item.model_dump() for item in y_trials_data], indent=2),
                    encoding="utf-8",
                )

    if skip_y:
        print("Skipped Y leakage eval (--skip-y). Run y-sweep on traces when ready.", flush=True)
        return run_id

    summary = build_summary(run_id, scaffolds, [pair.id for pair in pairs], x_results, leakage_results)
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print_summary(summary)
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

    mean_b50_scratch: dict[str, float | None] = {}
    mean_b50_with_trace: dict[str, float | None] = {}
    mean_leakage: dict[str, float | None] = {}
    leakage_by_pair: dict[str, dict[str, float | None]] = defaultdict(dict)

    for scaffold in scaffolds:
        scaffold_leakage = [r for r in leakage_results if r.scaffold == scaffold]
        mean_b50_scratch[scaffold] = _mean_or_none(
            [r.b50_scratch for r in scaffold_leakage if r.b50_scratch is not None]
        )
        mean_b50_with_trace[scaffold] = _mean_or_none(
            [r.b50_with_trace for r in scaffold_leakage if r.b50_with_trace is not None]
        )
        mean_leakage[scaffold] = _mean_or_none(
            [r.leakage for r in scaffold_leakage if r.leakage is not None]
        )
        for pair_id in pair_ids:
            pair_leakage = [r for r in scaffold_leakage if r.pair_id == pair_id and r.leakage is not None]
            if pair_leakage:
                leakage_by_pair[pair_id][scaffold] = mean(r.leakage for r in pair_leakage)

    return RunSummary(
        run_id=run_id,
        scaffolds=scaffolds,
        pair_ids=pair_ids,
        x_accuracy_by_scaffold=x_accuracy_by_scaffold,
        x_accuracy_by_pair=dict(x_accuracy_by_pair),
        mean_b50_scratch=mean_b50_scratch,
        mean_b50_with_trace=mean_b50_with_trace,
        mean_leakage=mean_leakage,
        leakage_by_pair=dict(leakage_by_pair),
    )


def analyze_run(run_id: str) -> RunSummary:
    run_dir = RUNS_DIR / run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    x_results: list[XAccuracyResult] = []
    leakage_results: list[LeakageResult] = []

    grader_client = create_client(manifest["grader_provider"], manifest["grader_model"])
    pairs = {pair.id: pair for pair in load_pairs(manifest["pair_ids"])}
    grader_temperature = manifest.get("grader_temperature", GRADER_TEMPERATURE)

    token_budgets = manifest.get("token_budgets", TOKEN_BUDGETS)

    for pair_id in manifest["pair_ids"]:
        for scaffold_name in manifest["scaffolds"]:
            scaffold_dir = run_dir / pair_id / scaffold_name
            if not scaffold_dir.exists():
                continue
            for result_path in sorted(scaffold_dir.glob("*.json")):
                if result_path.name.endswith("_y_trials.json"):
                    continue
                result = load_scaffold_result(result_path)
                pair = pairs[pair_id]
                x_results.append(
                    evaluate_x_accuracy(
                        pair,
                        scaffold_name,
                        result.trial,
                        result.answer_x,
                        grader_client,
                        grader_temperature,
                    )
                )
                y_path = result_path.with_name(f"{result.trial}_y_trials.json")
                if y_path.exists():
                    y_data = json.loads(y_path.read_text(encoding="utf-8"))
                    scratch = [item for item in y_data if item["condition"] == "scratch"]
                    with_trace = [item for item in y_data if item["condition"] == "with_trace"]
                    scratch_acc: dict[int, list[bool]] = defaultdict(list)
                    trace_acc: dict[int, list[bool]] = defaultdict(list)
                    for item in scratch:
                        scratch_acc[item["budget"]].append(item["correct"])
                    for item in with_trace:
                        trace_acc[item["budget"]].append(item["correct"])
                    b50_scratch = compute_b50(
                        {budget: mean(vals) for budget, vals in scratch_acc.items()},
                        token_budgets,
                    )
                    b50_trace = compute_b50(
                        {budget: mean(vals) for budget, vals in trace_acc.items()},
                        token_budgets,
                    )
                    leakage = None
                    undefined_reason = None
                    if b50_scratch.b50 is None:
                        undefined_reason = b50_scratch.undefined_reason
                    elif b50_trace.b50 is None:
                        undefined_reason = "B50 with trace undefined"
                    else:
                        leakage = (b50_scratch.b50 - b50_trace.b50) / b50_scratch.b50
                    leakage_results.append(
                        LeakageResult(
                            pair_id=pair_id,
                            scaffold=scaffold_name,
                            trial=result.trial,
                            b50_scratch=b50_scratch.b50,
                            b50_with_trace=b50_trace.b50,
                            leakage=leakage,
                            undefined_reason=undefined_reason,
                        )
                    )

    summary = build_summary(
        run_id,
        manifest["scaffolds"],
        manifest["pair_ids"],
        x_results,
        leakage_results,
    )
    (run_dir / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print_summary(summary)
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
    pair = {pair.id: pair for pair in load_pairs()}[result.pair_id]
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
    print(json.dumps(leakage_result.model_dump(), indent=2))
    return leakage_result


def print_summary(summary: RunSummary) -> None:
    headers = ["Scaffold", "X accuracy", "B50 scratch", "B50 w/ trace", "Leakage"]
    rows: list[list[str]] = []
    for scaffold in summary.scaffolds:
        rows.append(
            [
                scaffold,
                f"{summary.x_accuracy_by_scaffold.get(scaffold, 0.0):.2f}",
                _fmt(summary.mean_b50_scratch.get(scaffold)),
                _fmt(summary.mean_b50_with_trace.get(scaffold)),
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
    return f"{value:.1f}"
