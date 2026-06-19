from __future__ import annotations

import json
from pathlib import Path

from config import (
    DATA_DIR,
    FAST_TOKEN_BUDGETS,
    GENERATION_MAX_SPEND,
    GENERATION_MODEL,
    GENERATION_PROVIDER,
    GRADER_MODEL,
    GRADER_PROVIDER,
    SOLVER_MODEL,
    SOLVER_PROVIDER,
)
from generation.fewshots import load_selected_fewshots, run_stage0
from generation.generate import generate_candidates
from generation.measure import PairMetrics, measure_pairs
from generation.probe import probe_model
from generation.scoring import (
    FINAL_THRESHOLDS,
    FilterThresholds,
    is_trivially_derivable,
    passes,
    reasons,
    score,
)
from generation.spend import configure_spend, current_spend, reset_spend, spend_snapshot
from generation.verify import verify_pairs
from llm.factory import create_client
from models import QuestionPair

CAND_PATH = DATA_DIR / "_gen_candidates.json"
VERIFIED_PATH = DATA_DIR / "_gen_verified.json"
VERIFY_RECORDS_PATH = DATA_DIR / "_gen_verify_records.json"
MEASURED_PATH = DATA_DIR / "_gen_measured.json"

PAIR_KEYS = ("id", "x", "y", "answer_x", "answer_y", "relation", "y_proximity_signals")


def _save_pairs(path: Path, pairs: list[QuestionPair]) -> None:
    path.write_text(
        json.dumps([p.model_dump() for p in pairs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_pairs(path: Path) -> list[QuestionPair]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [QuestionPair.model_validate(item) for item in raw]


def _clean_pair_dict(pair: QuestionPair) -> dict:
    full = pair.model_dump()
    return {k: full[k] for k in PAIR_KEYS}


def run_pipeline(
    target_candidates: int = 160,
    n_per_call: int = 4,
    max_fewshots: int = 8,
    y_trials: int = 4,
    x_trials: int = 3,
    token_budgets: list[int] | None = None,
    max_spend: float = GENERATION_MAX_SPEND,
    max_final: int = 25,
    output_name: str = "pairs_generated_v1",
    verify_samples: int = 2,
    fresh: bool = False,
    refresh_fewshots: bool = False,
    thresholds: FilterThresholds = FINAL_THRESHOLDS,
) -> dict:
    budgets = token_budgets or FAST_TOKEN_BUDGETS
    configure_spend(max_spend)
    reset_spend()

    print(f"Probing generation model {GENERATION_PROVIDER}/{GENERATION_MODEL} ...", flush=True)
    reply = probe_model(GENERATION_PROVIDER, GENERATION_MODEL)
    print(f"  generation model OK ({reply!r})", flush=True)

    # Few-shots (Stage 0).
    fewshots = [] if refresh_fewshots else load_selected_fewshots()
    if not fewshots:
        print("No cached few-shots; running Stage 0 selection ...", flush=True)
        _, fewshots = run_stage0(
            y_trials=y_trials, x_trials=x_trials, token_budgets=budgets, max_fewshots=max_fewshots
        )
    print(f"Using {len(fewshots)} few-shots: {[p.id for p in fewshots]}", flush=True)

    gen_client = create_client(GENERATION_PROVIDER, GENERATION_MODEL)
    solver = create_client(SOLVER_PROVIDER, SOLVER_MODEL)
    grader = create_client(GRADER_PROVIDER, GRADER_MODEL)

    # Stage 1: generate.
    if not fresh and CAND_PATH.exists():
        candidates = _load_pairs(CAND_PATH)
        print(f"Stage 1: loaded {len(candidates)} cached candidates from {CAND_PATH.name}", flush=True)
    else:
        print(f"Stage 1: generating ~{target_candidates} candidates with {GENERATION_MODEL} ...", flush=True)
        pool_stems = [p.x for p in fewshots]
        candidates = generate_candidates(
            fewshots, target_candidates, n_per_call=n_per_call,
            seed_pool_stems=pool_stems, client=gen_client,
        )
        _save_pairs(CAND_PATH, candidates)
    print(f"  candidates: {len(candidates)} | spend so far ${current_spend():.2f}", flush=True)

    # Stage 2: verify.
    if not fresh and VERIFIED_PATH.exists():
        verified = _load_pairs(VERIFIED_PATH)
        print(f"Stage 2: loaded {len(verified)} cached verified pairs", flush=True)
    else:
        print(f"Stage 2: verifying {len(candidates)} candidates ({verify_samples} samples each) ...", flush=True)
        verified, records = verify_pairs(candidates, client=gen_client, samples=verify_samples)
        _save_pairs(VERIFIED_PATH, verified)
        VERIFY_RECORDS_PATH.write_text(
            json.dumps([r.__dict__ for r in records], indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"  verified: {len(verified)}/{len(candidates)} | spend so far ${current_spend():.2f}", flush=True)

    # Stage 3: measure leakage.
    if not fresh and MEASURED_PATH.exists():
        measured = [PairMetrics.from_dict(d) for d in json.loads(MEASURED_PATH.read_text(encoding="utf-8"))]
        print(f"Stage 3: loaded {len(measured)} cached measurements", flush=True)
    else:
        print(f"Stage 3: measuring leakage for {len(verified)} verified pairs ...", flush=True)
        measured = measure_pairs(
            verified, solver, grader,
            y_trials=y_trials, x_trials=x_trials, token_budgets=budgets,
            progress_label="measure",
        )
        MEASURED_PATH.write_text(
            json.dumps([m.to_dict() for m in measured], indent=2), encoding="utf-8"
        )
    print(f"  measured: {len(measured)} | spend so far ${current_spend():.2f}", flush=True)

    # Stage 4: filter, rank, write.
    by_id = {p.id: p for p in verified}
    ranked = sorted(measured, key=score, reverse=True)
    selected = []
    dropped_trivial: list[tuple[str, str]] = []
    for m in ranked:
        if not passes(m, thresholds):
            continue
        trivial, why = is_trivially_derivable(by_id[m.pair_id])
        if trivial:
            dropped_trivial.append((m.pair_id, why))
            continue
        selected.append(m)
        if len(selected) >= max_final:
            break
    if dropped_trivial:
        print(f"  dropped {len(dropped_trivial)} structurally-trivial pairs:", flush=True)
        for pid, why in dropped_trivial:
            print(f"    {pid}: {why}", flush=True)

    out_pairs = [_clean_pair_dict(by_id[m.pair_id]) for m in selected]
    out_path = DATA_DIR / f"{output_name}.json"
    out_path.write_text(json.dumps(out_pairs, indent=2, ensure_ascii=False), encoding="utf-8")

    metrics_path = DATA_DIR / f"{output_name}.metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "thresholds": thresholds.__dict__,
                "solver_model": SOLVER_MODEL,
                "generation_model": GENERATION_MODEL,
                "fewshots_used": [p.id for p in fewshots],
                "n_candidates": len(candidates),
                "n_verified": len(verified),
                "n_selected": len(selected),
                "dropped_trivially_derivable": [
                    {"pair_id": pid, "reason": why} for pid, why in dropped_trivial
                ],
                "spend": spend_snapshot(),
                "selected": [
                    {**_clean_pair_dict(by_id[m.pair_id]), "_metrics": m.to_dict(), "_score": score(m)}
                    for m in selected
                ],
                "all_measured": [
                    {
                        "pair_id": m.pair_id,
                        "metrics": m.to_dict(),
                        "score": score(m),
                        "passes": passes(m, thresholds),
                        "reasons": reasons(m, thresholds),
                    }
                    for m in ranked
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    _print_summary(selected, by_id, len(candidates), len(verified))
    print(f"\nWrote {len(out_pairs)} pairs -> {out_path}")
    print(f"Wrote metrics -> {metrics_path}")
    print(f"Total spend: ${current_spend():.2f} (cap ${max_spend:.2f})")

    return {
        "output": str(out_path),
        "metrics": str(metrics_path),
        "n_selected": len(selected),
        "spend": current_spend(),
    }


def _print_summary(selected, by_id, n_candidates, n_verified) -> None:
    print("\n=== Selected pairs (trace_leak / ans_leak / y_scratch / x_acc / score) ===")
    for m in selected:
        pair = by_id[m.pair_id]
        print(
            f"  {pair.id:<10} {m.trace_leakage:+.2f} / {m.answer_only_leakage:+.2f} / "
            f"{m.y_acc_scratch:.2f} / {m.x_scratch_acc:.2f} / {score(m):+.2f}  | {pair.x[:60]}"
        )
    print(f"\nCandidates: {n_candidates} -> verified: {n_verified} -> selected: {len(selected)}")
