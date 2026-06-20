from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from config import (
    DATA_DIR,
    FAST_TOKEN_BUDGETS,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    GRADER_MODEL,
    GRADER_PROVIDER,
    SOLVER_MODEL,
    SOLVER_PROVIDER,
    resolve_pairs_path,
)
from eval.y_leakage import normalized_leakage
from generation.measure import measure_pairs
from generation.scoring import FINAL_THRESHOLDS, is_trivially_derivable, passes, reasons, score
from generation.verify import verify_pairs
from llm.factory import create_client
from models import QuestionPair


def _load_pairs(path: Path) -> list[QuestionPair]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [QuestionPair.model_validate(item) for item in raw]


def vet_pairs_file(
    pairs_file: str | Path,
    y_trials: int = 4,
    x_trials: int = 3,
    token_budgets: list[int] | None = None,
    verify_samples: int = 2,
    trace_provider: str | None = GENERATION_PROVIDER,
    trace_model: str | None = GENERATION_MODEL,
    skip_verify: bool = False,
) -> dict:
    """Vet a hand-authored pairs file: confirm the gold answers re-solve, then
    measure leakage. The leaking X trace is produced by a strong model
    (trace_provider/trace_model) so hard X problems still yield a correct trace,
    while Y accuracy is measured with the cheap solver."""
    budgets = token_budgets or FAST_TOKEN_BUDGETS
    path = resolve_pairs_path(pairs_file)
    pairs = _load_pairs(path)
    print(f"Loaded {len(pairs)} pairs from {path}", flush=True)

    solver = create_client(SOLVER_PROVIDER, SOLVER_MODEL)
    grader = create_client(GRADER_PROVIDER, GRADER_MODEL)
    strong = create_client(GENERATION_PROVIDER, GENERATION_MODEL)

    trace_client = None
    use_strong_trace = trace_model is not None and not (
        trace_provider == SOLVER_PROVIDER and trace_model == SOLVER_MODEL
    )
    if use_strong_trace:
        trace_client = create_client(trace_provider, trace_model)
        print(f"Leaking X trace will be produced by {trace_provider}/{trace_model}", flush=True)
    else:
        print(f"Leaking X trace will be produced by the solver {SOLVER_PROVIDER}/{SOLVER_MODEL}", flush=True)

    verified_ids: set[str] = set()
    verify_lookup: dict[str, str] = {}
    if skip_verify:
        print("Skipping answer verification (--skip-verify).", flush=True)
        verified_ids = {p.id for p in pairs}
    else:
        print(f"Verifying {len(pairs)} pairs ({verify_samples} samples each) with {GENERATION_MODEL} ...", flush=True)
        accepted, records = verify_pairs(pairs, client=strong, samples=verify_samples)
        verified_ids = {p.id for p in accepted}
        verify_lookup = {r.pair_id: r.reason for r in records}
        print(f"  verified: {len(verified_ids)}/{len(pairs)}", flush=True)

    print(f"Measuring leakage for {len(pairs)} pairs ...", flush=True)
    measured = measure_pairs(
        pairs, solver, grader,
        y_trials=y_trials, x_trials=x_trials, token_budgets=budgets,
        progress_label="vet", trace_client=trace_client,
    )

    by_id = {p.id: p for p in pairs}
    report = _report(path, pairs, measured, by_id, verified_ids, verify_lookup)
    _print_report(report["pairs"], report["by_flavor"], verify_lookup)

    metrics_path = path.with_suffix(".vet.json")
    metrics_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote vet metrics -> {metrics_path}")
    return report


def _report(path, pairs, measured, by_id, verified_ids, verify_lookup) -> dict:
    rows = []
    for m in measured:
        pair = by_id[m.pair_id]
        trivial, why = is_trivially_derivable(pair)
        rows.append(
            {
                "pair_id": m.pair_id,
                "flavor": pair.flavor or "(none)",
                "verified": m.pair_id in verified_ids,
                "verify_reason": verify_lookup.get(m.pair_id, ""),
                "trace_leakage": m.trace_leakage,
                "trace_leakage_normalized": normalized_leakage(m.y_acc_scratch, m.y_acc_full_trace),
                "answer_only_leakage": m.answer_only_leakage,
                "y_acc_scratch": m.y_acc_scratch,
                "y_acc_full_trace": m.y_acc_full_trace,
                "x_scratch_acc": m.x_scratch_acc,
                "used_correct_trace": m.used_correct_trace,
                "score": score(m),
                "passes_final": passes(m, FINAL_THRESHOLDS) and not trivial,
                "fail_reasons": reasons(m, FINAL_THRESHOLDS) + (["structurally trivial: " + why] if trivial else []),
            }
        )
    return {
        "pairs_file": str(path),
        "n_pairs": len(pairs),
        "n_verified": len(verified_ids),
        "thresholds": FINAL_THRESHOLDS.__dict__,
        "by_flavor": _by_flavor_summary(rows),
        "pairs": rows,
    }


def _by_flavor_summary(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["flavor"]].append(r)
    out = {}
    for flavor, items in groups.items():
        n = len(items)
        mean_scratch = sum(i["y_acc_scratch"] for i in items) / n
        mean_trace = sum(i["y_acc_full_trace"] for i in items) / n
        out[flavor] = {
            "n": n,
            "mean_trace_leakage": sum(i["trace_leakage"] for i in items) / n,
            "mean_trace_leakage_normalized": normalized_leakage(mean_scratch, mean_trace),
            "mean_answer_only_leakage": sum(i["answer_only_leakage"] for i in items) / n,
            "mean_y_acc_scratch": mean_scratch,
            "mean_x_scratch_acc": sum(i["x_scratch_acc"] for i in items) / n,
            "n_passes_final": sum(1 for i in items if i["passes_final"]),
        }
    return out


def _fmt_opt(value: float | None) -> str:
    return "  n/a" if value is None else f"{value:+.2f}"


def _print_report(rows: list[dict], summary: dict, verify_lookup: dict) -> None:
    print(
        "\n=== Per-pair (trace_leak / trace_leak_norm / ans_leak / y_scratch / x_acc(cheap) / pass) ===",
        flush=True,
    )
    by_flavor: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_flavor[r["flavor"]].append(r)

    for flavor in sorted(by_flavor):
        print(f"\n[{flavor}]", flush=True)
        for r in by_flavor[flavor]:
            ver = "ok " if r["verified"] else "BAD"
            print(
                f"  {r['pair_id']:<10} ver={ver} "
                f"{r['trace_leakage']:+.2f} / {_fmt_opt(r['trace_leakage_normalized'])} / "
                f"{r['answer_only_leakage']:+.2f} / "
                f"{r['y_acc_scratch']:.2f} / {r['x_scratch_acc']:.2f} / "
                f"{'PASS' if r['passes_final'] else 'fail'} "
                f"{'(correct-trace)' if r['used_correct_trace'] else '(no-correct-trace)'}",
                flush=True,
            )
            if not r["verified"]:
                print(f"             verify: {verify_lookup.get(r['pair_id'], 'n/a')}", flush=True)
            if r["fail_reasons"]:
                print(f"             why-fail: {'; '.join(r['fail_reasons'])}", flush=True)

    print(
        "\n=== By flavor (mean trace_leak / mean trace_leak_norm / mean ans_leak / mean y_scratch / passes) ===",
        flush=True,
    )
    for flavor in sorted(summary):
        s = summary[flavor]
        print(
            f"  {flavor:<18} n={s['n']}  "
            f"{s['mean_trace_leakage']:+.2f} / {_fmt_opt(s['mean_trace_leakage_normalized'])} / "
            f"{s['mean_answer_only_leakage']:+.2f} / "
            f"{s['mean_y_acc_scratch']:.2f}  passes={s['n_passes_final']}/{s['n']}",
            flush=True,
        )
