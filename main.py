from __future__ import annotations

import argparse
from pathlib import Path

from config import (
    DEFAULT_JOB_WORKERS,
    DEFAULT_MODEL,
    DEFAULT_PROVIDER,
    DEFAULT_TEMPERATURE,
    DEFAULT_TRIALS,
    DEFAULT_WORKERS,
    DEFAULT_Y_TRIALS,
    FAST_SCAFFOLDS,
    FAST_TOKEN_BUDGETS,
    GRADER_TEMPERATURE,
    RUNS_DIR,
)
from eval.runner import analyze_run, run_full_eval, y_sweep_trace
from visualizer.server import serve


def _parse_csv(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thread Cutter isolated reasoning eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run scaffolds and full eval")
    run_parser.add_argument(
        "--scaffolds",
        default="control,baseline_avoid,threaded,two_agent",
        help="Comma-separated scaffold names",
    )
    run_parser.add_argument("--pairs", default="all", help="Comma-separated pair ids or 'all'")
    run_parser.add_argument(
        "--pairs-file",
        default=None,
        help="Path to question pairs JSON (default: data/pairs.json)",
    )
    run_parser.add_argument("--trials", type=int, default=None)
    run_parser.add_argument("--y-trials", type=int, default=None)
    run_parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    run_parser.add_argument("--model", default=DEFAULT_MODEL)
    run_parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature for scaffolds and Y answers",
    )
    run_parser.add_argument(
        "--grader-temperature",
        type=float,
        default=GRADER_TEMPERATURE,
        help="Sampling temperature for LLM grader calls",
    )
    run_parser.add_argument("--run-id", default=None)
    run_parser.add_argument(
        "--fast",
        action="store_true",
        help="Quick iteration: 1 trial, 2 y-trials, 6 budgets, control+baseline only",
    )
    run_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel Y API calls per sweep (default: 0 = all budgets×trials at once)",
    )
    run_parser.add_argument(
        "--job-workers",
        type=int,
        default=None,
        help="Parallel top-level tasks (default: 0 = unlimited)",
    )
    run_parser.add_argument(
        "--budgets",
        default=None,
        help="Comma-separated token budgets for Y sweep (default: 9 full or 6 in --fast)",
    )
    run_parser.add_argument(
        "--skip-y",
        action="store_true",
        help="Only run scaffolds on X; skip Y leakage eval",
    )

    analyze_parser = subparsers.add_parser("analyze", help="Recompute summary for a run")
    analyze_parser.add_argument("--run-id", required=True)
    analyze_parser.add_argument(
        "--job-workers",
        type=int,
        default=None,
        help="Parallel trace regrades (default: 0 = unlimited)",
    )

    sweep_parser = subparsers.add_parser("y-sweep", help="Run Y leakage sweep for one trace")
    sweep_parser.add_argument("--trace", required=True, help="Path to scaffold result JSON")
    sweep_parser.add_argument("--y-trials", type=int, default=DEFAULT_Y_TRIALS)
    sweep_parser.add_argument("--provider", default=DEFAULT_PROVIDER)
    sweep_parser.add_argument("--model", default=DEFAULT_MODEL)
    sweep_parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    sweep_parser.add_argument("--grader-temperature", type=float, default=GRADER_TEMPERATURE)
    sweep_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    sweep_parser.add_argument("--budgets", default=None)
    sweep_parser.add_argument(
        "--run-id",
        default=None,
        help="Run id for scratch baseline cache lookup",
    )

    viz_parser = subparsers.add_parser("visualize", help="Launch trace visualizer web UI")
    viz_parser.add_argument("--host", default="127.0.0.1")
    viz_parser.add_argument("--port", type=int, default=8765)

    return parser


def _resolve_run_options(args) -> dict:
    fast = getattr(args, "fast", False)
    scaffolds = _parse_csv(args.scaffolds)
    if fast:
        scaffolds = list(FAST_SCAFFOLDS)

    trials = args.trials if args.trials is not None else (1 if fast else DEFAULT_TRIALS)
    y_trials = args.y_trials if args.y_trials is not None else (2 if fast else DEFAULT_Y_TRIALS)
    workers = args.workers if args.workers is not None else DEFAULT_WORKERS
    job_workers = args.job_workers if args.job_workers is not None else DEFAULT_JOB_WORKERS
    token_budgets = (
        _parse_int_csv(args.budgets)
        if args.budgets
        else (FAST_TOKEN_BUDGETS if fast else None)
    )

    return {
        "scaffolds": scaffolds,
        "trials": trials,
        "y_trials": y_trials,
        "workers": workers,
        "job_workers": job_workers,
        "token_budgets": token_budgets,
        "skip_y": getattr(args, "skip_y", False),
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        pair_ids = _parse_csv(args.pairs)
        opts = _resolve_run_options(args)
        run_id = run_full_eval(
            scaffolds=opts["scaffolds"],
            pair_ids=pair_ids or None,
            trials=opts["trials"],
            y_trials=opts["y_trials"],
            provider=args.provider,
            model=args.model,
            run_id=args.run_id,
            token_budgets=opts["token_budgets"],
            workers=opts["workers"],
            job_workers=opts["job_workers"],
            skip_y=opts["skip_y"],
            temperature=args.temperature,
            grader_temperature=args.grader_temperature,
            pairs_file=args.pairs_file,
        )
        print(f"Run complete: {run_id}")
        return

    if args.command == "analyze":
        job_workers = args.job_workers if args.job_workers is not None else DEFAULT_JOB_WORKERS
        analyze_run(args.run_id, job_workers=job_workers)
        return

    if args.command == "y-sweep":
        token_budgets = _parse_int_csv(args.budgets) if args.budgets else None
        run_dir = RUNS_DIR / args.run_id if args.run_id else None
        y_sweep_trace(
            trace_path=Path(args.trace),
            provider=args.provider,
            model=args.model,
            y_trials=args.y_trials,
            token_budgets=token_budgets,
            workers=args.workers,
            run_dir=run_dir,
            temperature=args.temperature,
            grader_temperature=args.grader_temperature,
        )
        return

    if args.command == "visualize":
        serve(host=args.host, port=args.port)
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
