from __future__ import annotations

import re
from dataclasses import dataclass

from generation.measure import PairMetrics
from models import QuestionPair


@dataclass
class FilterThresholds:
    max_y_acc_scratch: float = 0.45
    min_trace_leakage: float = 0.30
    max_answer_only_leakage: float = 0.12
    min_x_scratch_acc: float = 0.0  # X must be solvable enough to produce a real trace
    max_x_scratch_acc: float = 1.01  # (set < 1 to require X be non-trivial)
    require_correct_trace: bool = False  # leaking trace must come from a correct X solution


# Looser bar for picking few-shot exemplars: we just need good illustrations.
FEWSHOT_THRESHOLDS = FilterThresholds(
    max_y_acc_scratch=0.65,
    min_trace_leakage=0.20,
    max_answer_only_leakage=0.20,
)

# Stricter bar for the final curated output set.
FINAL_THRESHOLDS = FilterThresholds(
    max_y_acc_scratch=0.45,
    min_trace_leakage=0.30,
    max_answer_only_leakage=0.12,
    require_correct_trace=True,
)


def _to_number(text: str) -> float | None:
    text = text.strip()
    m = re.fullmatch(r"(-?\d+)\s*/\s*(-?\d+)", text)
    if m:
        denom = float(m.group(2))
        return float(m.group(1)) / denom if denom else None
    try:
        return float(text)
    except ValueError:
        return None


def is_trivially_derivable(pair: QuestionPair) -> tuple[bool, str]:
    """Structural guard: True if Y is obtainable from X's bare answer using only
    numbers already visible in Y's text (the answer-only condition hides X's
    question, so such pairs leak via X's answer and are low quality)."""
    ax = pair.answer_x.strip()
    ay = pair.answer_y.strip()

    # X answer is a list/sequence of values -> exposes terms Y likely needs.
    parts = [p for p in ax.split(",") if p.strip()]
    if len(parts) >= 3 and all(_to_number(p) is not None for p in parts):
        return True, "X answer is a list/sequence that exposes Y"

    nax = _to_number(ax)
    nay = _to_number(ay)
    if nax is not None and nay is not None:
        y_ints = [int(t) for t in re.findall(r"-?\d+", pair.y)]
        for total in y_ints:
            # complementary count / sums to a constant present in Y's text
            if abs((nax + nay) - total) < 1e-9 and total > max(abs(nax), abs(nay)):
                return True, f"complement: answer_x + answer_y = {total} (a number in Y's text)"
    return False, ""


def score(metrics: PairMetrics) -> float:
    """Higher is better: reward trace leakage, penalize answer-only leakage,
    and reward Y being hard from scratch."""
    return (
        metrics.trace_leakage
        - metrics.answer_only_leakage
        + 0.25 * (1.0 - metrics.y_acc_scratch)
    )


def passes(metrics: PairMetrics, thresholds: FilterThresholds) -> bool:
    return (
        metrics.y_acc_scratch <= thresholds.max_y_acc_scratch
        and metrics.trace_leakage >= thresholds.min_trace_leakage
        and metrics.answer_only_leakage <= thresholds.max_answer_only_leakage
        and thresholds.min_x_scratch_acc <= metrics.x_scratch_acc <= thresholds.max_x_scratch_acc
        and (not thresholds.require_correct_trace or metrics.used_correct_trace)
    )


def reasons(metrics: PairMetrics, thresholds: FilterThresholds) -> list[str]:
    out: list[str] = []
    if metrics.y_acc_scratch > thresholds.max_y_acc_scratch:
        out.append(f"Y too easy from scratch ({metrics.y_acc_scratch:.2f})")
    if metrics.trace_leakage < thresholds.min_trace_leakage:
        out.append(f"trace leakage too low ({metrics.trace_leakage:.2f})")
    if metrics.answer_only_leakage > thresholds.max_answer_only_leakage:
        out.append(f"answer-only leakage too high ({metrics.answer_only_leakage:.2f})")
    if not (thresholds.min_x_scratch_acc <= metrics.x_scratch_acc <= thresholds.max_x_scratch_acc):
        out.append(f"X solvability out of range ({metrics.x_scratch_acc:.2f})")
    if thresholds.require_correct_trace and not metrics.used_correct_trace:
        out.append("no correct X trace available (leak measured from wrong solution)")
    return out
