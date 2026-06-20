from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QuestionPair(BaseModel):
    id: str
    x: str
    y: str
    answer_x: str
    answer_y: str
    relation: str
    y_proximity_signals: list[str] = Field(default_factory=list)
    # Optional metadata (default empty for backward compatibility with existing
    # pairs files). flavor tags the difficulty/style; rubric_* and gold_solution
    # support a future move to rubric / LLM-judge grading without reshaping pairs.
    flavor: str = ""
    rubric_x: str = ""
    rubric_y: str = ""
    gold_solution: str = ""


class TraceStep(BaseModel):
    role: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScaffoldResult(BaseModel):
    pair_id: str
    scaffold: str
    trial: int
    answer_x: str
    trace: list[TraceStep]
    metadata: dict[str, Any] = Field(default_factory=dict)


class GradeResult(BaseModel):
    correct: bool
    method: str
    detail: str = ""


class YTrialResult(BaseModel):
    budget: int
    condition: str
    trial: int
    answer: str
    correct: bool
    grade_method: str = ""
    grade_detail: str = ""
    raw_response: str = ""
    prompt_messages: list[dict[str, str]] = Field(default_factory=list)


class B50Result(BaseModel):
    b50: float | None
    accuracies: dict[int, float]
    undefined_reason: str | None = None


class LeakageResult(BaseModel):
    pair_id: str
    scaffold: str
    trial: int
    y_accuracy_scratch: float
    y_accuracy_with_trace: float
    leakage: float
    x_correct: bool = True


class XAccuracyResult(BaseModel):
    pair_id: str
    scaffold: str
    trial: int
    answer_x: str
    correct: bool


class RunSummary(BaseModel):
    run_id: str
    scaffolds: list[str]
    pair_ids: list[str]
    x_accuracy_by_scaffold: dict[str, float]
    x_accuracy_by_pair: dict[str, dict[str, float]]
    mean_y_accuracy_scratch: dict[str, float | None]
    mean_y_accuracy_with_trace: dict[str, float | None]
    mean_leakage: dict[str, float | None]
    leakage_by_pair: dict[str, dict[str, float | None]]
    # Leakage normalized by headroom: (Y_trace - Y_scratch) / (1 - Y_scratch),
    # computed from aggregate mean accuracies. None when there is no headroom.
    mean_leakage_normalized: dict[str, float | None] = Field(default_factory=dict)
    leakage_normalized_by_pair: dict[str, dict[str, float | None]] = Field(default_factory=dict)
    leakage_trials_used: dict[str, int] = Field(default_factory=dict)
    leakage_trials_total: dict[str, int] = Field(default_factory=dict)
