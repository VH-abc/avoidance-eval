from __future__ import annotations

from config import GRADER_TEMPERATURE
from llm.client import LLMClient
from models import QuestionPair, XAccuracyResult
from eval.grader import grade_x_answer


def evaluate_x_accuracy(
    pair: QuestionPair,
    scaffold: str,
    trial: int,
    answer_x: str,
    grader_client: LLMClient,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> XAccuracyResult:
    grade = grade_x_answer(pair, answer_x, grader_client, grader_temperature)
    return XAccuracyResult(
        pair_id=pair.id,
        scaffold=scaffold,
        trial=trial,
        answer_x=answer_x,
        correct=grade.correct,
    )
