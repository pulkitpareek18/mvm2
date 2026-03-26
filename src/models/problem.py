from __future__ import annotations

from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


class ProblemType(str, Enum):
    ARITHMETIC = "arithmetic"
    ALGEBRA = "algebra"
    GEOMETRY = "geometry"
    WORD_PROBLEM = "word_problem"
    NUMBER_THEORY = "number_theory"
    STATISTICS = "statistics"
    CALCULUS = "calculus"
    LINEAR_ALGEBRA = "linear_algebra"
    DIFFERENTIAL_EQUATIONS = "differential_equations"
    TRIGONOMETRY = "trigonometry"
    PROBABILITY = "probability"
    DISCRETE_MATH = "discrete_math"
    COMPETITION = "competition"
    UNKNOWN = "unknown"


class MathProblem(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    raw_text: str
    normalized_text: str | None = None
    problem_type: ProblemType = ProblemType.UNKNOWN
    image_data: bytes | None = None
