from __future__ import annotations

from pydantic import BaseModel


class SolutionStep(BaseModel):
    step_number: int
    description: str
    mathematical_expression: str | None = None
    result: str | None = None


class ModelSolution(BaseModel):
    model_name: str
    steps: list[SolutionStep] = []
    final_answer: str = ""
    raw_response: str = ""
    latency_ms: float = 0.0
    error: str | None = None
