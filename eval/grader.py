from __future__ import annotations

import re
from fractions import Fraction
from typing import Protocol

from config import GRADER_TEMPERATURE
from llm.client import LLMClient, Message
from models import GradeResult, QuestionPair
from scaffolds.base import extract_final_answer, has_final_answer


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _parse_fraction_value(text: str) -> float | None:
    match = re.search(r"(-?\d+)\s*/\s*(-?\d+)", text)
    if not match:
        return None
    denominator = int(match.group(2))
    if denominator == 0:
        return None
    return float(Fraction(int(match.group(1)), denominator))


def _parse_latex_frac_value(text: str) -> float | None:
    match = re.search(r"\\(?:d|t)?frac\{(-?\d+)\}\{(-?\d+)\}", text)
    if not match:
        return None
    denominator = int(match.group(2))
    if denominator == 0:
        return None
    return float(Fraction(int(match.group(1)), denominator))


def _parse_decimal_value(text: str) -> float | None:
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if len(matches) != 1:
        return None
    return float(matches[0])


def _parse_numeric_value(text: str) -> float | None:
    slash = _parse_fraction_value(text)
    if slash is not None:
        return slash
    latex = _parse_latex_frac_value(text)
    if latex is not None:
        return latex
    if "\\frac" in text:
        return None
    return _parse_decimal_value(text)


def _equivalent_value(candidate: str, gold: str) -> bool:
    gold_value = _parse_numeric_value(gold)
    candidate_value = _parse_numeric_value(candidate)
    if gold_value is not None and candidate_value is not None:
        return abs(gold_value - candidate_value) < 1e-9
    return False


def _rule_match(extracted: str, gold: str) -> bool:
    if _normalize(extracted) == _normalize(gold):
        return True
    return _equivalent_value(extracted, gold)


def answers_equivalent(a: str, b: str) -> bool:
    """Public helper: True if two answer strings are equivalent under the same
    normalization / numeric rules the grader uses."""
    return _rule_match(a, b)


def _unambiguous_answer_match(candidate: str, gold: str) -> bool:
    stripped = candidate.strip()
    if not stripped:
        return False
    if _rule_match(stripped, gold):
        return True
    return _equivalent_value(stripped, gold)


def _parse_llm_verdict(content: str) -> bool:
    normalized = re.sub(r"[^A-Z]", "", content.strip().upper())
    return normalized == "CORRECT"


class Grader(Protocol):
    """Pluggable grading strategy. Implementations decide whether a candidate
    answer is correct given the question, the gold answer, and (optionally) the
    full pair (for rubric-based grading)."""

    def grade(
        self,
        question: str,
        gold: str,
        candidate: str,
        *,
        pair: QuestionPair | None = None,
        require_final_answer: bool = True,
    ) -> GradeResult: ...


class ExactAnswerGrader:
    """Grades against a single checkable gold answer: fast deterministic rules
    (normalization, numeric/fraction equivalence) first, then a strict one-word
    LLM verdict as a fallback. This is the current default behavior."""

    def __init__(
        self,
        grader_client: LLMClient,
        grader_temperature: float = GRADER_TEMPERATURE,
    ) -> None:
        self.grader_client = grader_client
        self.grader_temperature = grader_temperature

    def grade(
        self,
        question: str,
        gold: str,
        candidate: str,
        *,
        pair: QuestionPair | None = None,
        require_final_answer: bool = True,
    ) -> GradeResult:
        if require_final_answer and not has_final_answer(candidate):
            if _unambiguous_answer_match(candidate, gold):
                return GradeResult(
                    correct=True,
                    method="rule",
                    detail="answer without prefix",
                )
            return GradeResult(
                correct=False,
                method="rule",
                detail="missing Final answer",
            )

        extracted = extract_final_answer(candidate)
        if _rule_match(extracted, gold):
            return GradeResult(correct=True, method="rule", detail="normalized match")

        prompt = (
            "You are a strict answer grader. The candidate must state the same final result "
            "as the gold answer.\n"
            "Reply CORRECT only if the candidate's final answer is mathematically or "
            "semantically equivalent to the gold answer.\n"
            "Reply INCORRECT if the candidate gives setup, reasoning, enumeration, a truncated "
            "response, or a wrong/incomplete final result. Do not give partial credit.\n"
            "Reply with exactly one word: CORRECT or INCORRECT.\n\n"
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"Candidate final answer: {extracted}"
        )
        completion = self.grader_client.complete(
            messages=[Message(role="user", content=prompt)],
            max_tokens=8,
            temperature=self.grader_temperature,
        )
        verdict = completion.content.strip()
        correct = _parse_llm_verdict(verdict)
        return GradeResult(correct=correct, method="llm", detail=verdict)


class RubricGrader:
    """Stub for future open-ended (option C) grading: judges a candidate against
    a natural-language rubric (`pair.rubric_x` / `pair.rubric_y`) via an LLM
    judge. Not wired into the default path yet."""

    def __init__(
        self,
        grader_client: LLMClient,
        grader_temperature: float = GRADER_TEMPERATURE,
    ) -> None:
        self.grader_client = grader_client
        self.grader_temperature = grader_temperature

    def grade(
        self,
        question: str,
        gold: str,
        candidate: str,
        *,
        pair: QuestionPair | None = None,
        require_final_answer: bool = True,
    ) -> GradeResult:
        raise NotImplementedError(
            "RubricGrader is a placeholder for open-ended grading; populate "
            "pair.rubric_x / pair.rubric_y and implement the judge before use."
        )


def make_grader(
    grader_client: LLMClient,
    grader_temperature: float = GRADER_TEMPERATURE,
    kind: str = "exact",
) -> Grader:
    """Single place that decides which grader to use. Switching the project to
    open-ended grading later means changing the default `kind` here (or threading
    it through), not the call sites."""
    if kind == "exact":
        return ExactAnswerGrader(grader_client, grader_temperature)
    if kind == "rubric":
        return RubricGrader(grader_client, grader_temperature)
    raise ValueError(f"Unknown grader kind: {kind}")


def grade_answer(
    question: str,
    gold: str,
    candidate: str,
    grader_client: LLMClient,
    grader_temperature: float = GRADER_TEMPERATURE,
    require_final_answer: bool = True,
) -> GradeResult:
    grader = make_grader(grader_client, grader_temperature)
    return grader.grade(
        question, gold, candidate, require_final_answer=require_final_answer
    )


def grade_x_answer(
    pair: QuestionPair,
    candidate: str,
    grader_client: LLMClient,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> GradeResult:
    grader = make_grader(grader_client, grader_temperature)
    return grader.grade(
        pair.x,
        pair.answer_x,
        candidate,
        pair=pair,
        require_final_answer=False,
    )


def grade_y_answer(
    pair: QuestionPair,
    candidate: str,
    grader_client: LLMClient,
    grader_temperature: float = GRADER_TEMPERATURE,
) -> GradeResult:
    grader = make_grader(grader_client, grader_temperature)
    return grader.grade(
        pair.y,
        pair.answer_y,
        candidate,
        pair=pair,
        require_final_answer=True,
    )
