from __future__ import annotations

from pydantic import BaseModel

from .solution import SolutionStep


class StepAlignment(BaseModel):
    canonical_step_number: int
    description: str
    model_steps: dict[str, SolutionStep | None] = {}  # model_name -> step (None if missing)
    agreement_ratio: float = 0.0  # 0.0 to 1.0
    symbolic_verified: bool | None = None  # None = not checked / inconclusive
    flagged: bool = False


class VerificationResult(BaseModel):
    problem_id: str
    problem_text: str
    model_solutions: dict[str, list[dict]] = {}  # model_name -> serialized steps
    aligned_steps: list[StepAlignment] = []
    final_answer: str = ""
    confidence: float = 0.0
    answer_agreement: dict[str, list[str]] = {}  # answer -> [model_names]
    debate_rounds: int = 0
    symbolic_override: bool = False
    audit_trail: list[str] = []
