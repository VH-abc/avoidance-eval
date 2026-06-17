from __future__ import annotations

import re

from config import GRADER_TEMPERATURE
from llm.client import LLMClient, Message
from models import GradeResult, QuestionPair
from scaffolds.base import extract_final_answer


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _numeric_match(candidate: str, gold: str) -> bool:
    candidate_nums = re.findall(r"-?\d+(?:\.\d+)?", candidate)
    gold_nums = re.findall(r"-?\d+(?:\.\d+)?", gold)
    if not gold_nums:
        return False
    return any(num in candidate_nums for num in gold_nums)


def _exact_or_normalized_match(candidate: str, gold: str) -> bool:
    if _normalize(candidate) == _normalize(gold):
        return True
    if gold.lower() in candidate.lower():
        return True
    if candidate.lower() in gold.lower() and len(candidate.strip()) > 0:
        return True
    return _numeric_match(candidate, gold)


def grade_answer(
    question: str,
    gold: str,
    candidate: str,
    grader_client: LLMClient,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> GradeResult:
    candidate = extract_final_answer(candidate)
    if _exact_or_normalized_match(candidate, gold):
        return GradeResult(correct=True, method="rule", detail="normalized match")

    prompt = (
        "You are a strict answer grader. Decide if the candidate answer is semantically correct "
        "for the question, given the gold answer.\n"
        "Reply with exactly one word: CORRECT or INCORRECT.\n\n"
        f"Question: {question}\n"
        f"Gold answer: {gold}\n"
        f"Candidate answer: {candidate}"
    )
    completion = grader_client.complete(
        messages=[Message(role="user", content=prompt)],
        max_tokens=8,
        temperature=grader_temperature,
    )
    verdict = completion.content.strip().upper()
    correct = verdict.startswith("CORRECT")
    return GradeResult(correct=correct, method="llm", detail=verdict)


def grade_x_answer(
    pair: QuestionPair,
    candidate: str,
    grader_client: LLMClient,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> GradeResult:
    return grade_answer(pair.x, pair.answer_x, candidate, grader_client, grader_temperature)


def grade_y_answer(
    pair: QuestionPair,
    candidate: str,
    grader_client: LLMClient,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> GradeResult:
    return grade_answer(pair.y, pair.answer_y, candidate, grader_client, grader_temperature)
