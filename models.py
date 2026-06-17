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


class B50Result(BaseModel):
    b50: float | None
    accuracies: dict[int, float]
    undefined_reason: str | None = None


class LeakageResult(BaseModel):
    pair_id: str
    scaffold: str
    trial: int
    b50_scratch: float | None
    b50_with_trace: float | None
    leakage: float | None
    undefined_reason: str | None = None


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
    mean_b50_scratch: dict[str, float | None]
    mean_b50_with_trace: dict[str, float | None]
    mean_leakage: dict[str, float | None]
    leakage_by_pair: dict[str, dict[str, float | None]]
