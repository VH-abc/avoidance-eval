from __future__ import annotations

from llm.client import LLMClient
from models import QuestionPair, ScaffoldResult, TraceStep
from scaffolds.control import run_control


def run_control_answer_only(
    pair: QuestionPair, client: LLMClient, trial: int, temperature: float = 1.0
) -> ScaffoldResult:
    result = run_control(pair, client, trial, temperature)
    return ScaffoldResult(
        pair_id=result.pair_id,
        scaffold="control_answer_only",
        trial=result.trial,
        answer_x=result.answer_x,
        trace=[
            TraceStep(
                role="assistant",
                content=f"Final answer: {result.answer_x}",
                metadata={"label": "x_answer_only", "answer_only": True},
            )
        ],
    )
