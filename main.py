from __future__ import annotations

import argparse
import os
import sys
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
    GENERATION_MAX_SPEND,
    GRADER_TEMPERATURE,
    RUNS_DIR,
)
from eval.runner import analyze_run, run_full_eval, y_sweep_trace
from visualizer.server import serve


DEFAULT_SCAFFOLDS = "control,control_answer_only,baseline_avoid,cheating_avoid,threaded,two_agent"


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
        default=None,
        help=(
            "Comma-separated scaffold names. Overrides --fast's scaffold set when given. "
            f"Default: {DEFAULT_SCAFFOLDS}"
        ),
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
        "--x-provider",
        default=None,
        help="Provider for solving X (default: same as --provider)",
    )
    run_parser.add_argument(
        "--x-model",
        default=None,
        help=(
            "Model that solves X in the scaffolds, letting X use a smarter model than Y. "
            "If unset, X uses --model (same model for both X and Y)."
        ),
    )
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

    vet_parser = subparsers.add_parser(
        "vet", help="Verify gold answers and measure leakage for a hand-authored pairs file"
    )
    vet_parser.add_argument(
        "--pairs-file", required=True, help="Path to question pairs JSON to vet"
    )
    vet_parser.add_argument("--y-trials", type=int, default=4)
    vet_parser.add_argument("--x-trials", type=int, default=3)
    vet_parser.add_argument("--verify-samples", type=int, default=2)
    vet_parser.add_argument("--budgets", default=None, help="Comma-separated Y token budgets")
    vet_parser.add_argument(
        "--trace-model",
        default=None,
        help=(
            "Model that produces the leaking X trace (default: the strong generation "
            "model). Pass 'solver' to use the cheap solver instead."
        ),
    )
    vet_parser.add_argument(
        "--trace-provider",
        default=None,
        help="Provider for --trace-model (default: the generation provider)",
    )
    vet_parser.add_argument(
        "--skip-verify", action="store_true", help="Skip the answer re-solve verification step"
    )

    gen_parser = subparsers.add_parser(
        "generate", help="Generate (X, Y) pairs with high trace-leakage, low answer-only-leakage"
    )
    gen_parser.add_argument("--target", type=int, default=160, help="Candidate pairs to generate")
    gen_parser.add_argument("--n-per-call", type=int, default=4, help="Candidates per generation call")
    gen_parser.add_argument("--max-fewshots", type=int, default=8)
    gen_parser.add_argument("--y-trials", type=int, default=4)
    gen_parser.add_argument("--x-trials", type=int, default=3)
    gen_parser.add_argument("--verify-samples", type=int, default=2)
    gen_parser.add_argument("--budgets", default=None, help="Comma-separated Y token budgets")
    gen_parser.add_argument("--max-spend", type=float, default=GENERATION_MAX_SPEND)
    gen_parser.add_argument("--max-final", type=int, default=25, help="Max curated pairs to output")
    gen_parser.add_argument("--output", default="pairs_generated_v1", help="Output basename in data/")
    gen_parser.add_argument("--gen-model", default=None, help="Override generation model string")
    gen_parser.add_argument("--fresh", action="store_true", help="Ignore cached stage outputs")
    gen_parser.add_argument(
        "--refresh-fewshots", action="store_true", help="Re-run Stage 0 few-shot selection"
    )

    return parser


def _resolve_run_options(args) -> dict:
    fast = getattr(args, "fast", False)
    if args.scaffolds is not None:
        # Explicit --scaffolds always wins, even alongside --fast.
        scaffolds = _parse_csv(args.scaffolds)
    elif fast:
        scaffolds = list(FAST_SCAFFOLDS)
    else:
        scaffolds = _parse_csv(DEFAULT_SCAFFOLDS)

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
    # Windows consoles default to cp1252; math symbols (e.g. the congruence sign)
    # in problem text would otherwise crash on print. Make stdout tolerant.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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
            x_provider=args.x_provider,
            x_model=args.x_model,
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

    if args.command == "vet":
        from config import GENERATION_PROVIDER, SOLVER_MODEL, SOLVER_PROVIDER
        from generation.vet import vet_pairs_file

        token_budgets = _parse_int_csv(args.budgets) if args.budgets else None
        if args.trace_model is not None and args.trace_model.lower() == "solver":
            trace_provider = SOLVER_PROVIDER
            trace_model = SOLVER_MODEL
        else:
            from config import GENERATION_MODEL

            trace_model = args.trace_model or GENERATION_MODEL
            trace_provider = args.trace_provider or GENERATION_PROVIDER
        vet_pairs_file(
            pairs_file=args.pairs_file,
            y_trials=args.y_trials,
            x_trials=args.x_trials,
            token_budgets=token_budgets,
            verify_samples=args.verify_samples,
            trace_provider=trace_provider,
            trace_model=trace_model,
            skip_verify=args.skip_verify,
        )
        return

    if args.command == "generate":
        if args.gen_model:
            os.environ["GENERATION_MODEL"] = args.gen_model
            import importlib

            import config as _config

            importlib.reload(_config)
        from generation.pipeline import run_pipeline

        token_budgets = _parse_int_csv(args.budgets) if args.budgets else None
        run_pipeline(
            target_candidates=args.target,
            n_per_call=args.n_per_call,
            max_fewshots=args.max_fewshots,
            y_trials=args.y_trials,
            x_trials=args.x_trials,
            token_budgets=token_budgets,
            max_spend=args.max_spend,
            max_final=args.max_final,
            output_name=args.output,
            verify_samples=args.verify_samples,
            fresh=args.fresh,
            refresh_fewshots=args.refresh_fewshots,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
