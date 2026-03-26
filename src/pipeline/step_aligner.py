from __future__ import annotations

import structlog

from ..models.solution import ModelSolution, SolutionStep
from ..models.verification import StepAlignment

logger = structlog.get_logger()


def align_steps(solutions: list[ModelSolution]) -> list[StepAlignment]:
    """Align steps across models using positional matching (v1: step N ≈ step N).

    For v1, we use the model with the most steps as the reference and map
    other models' steps by position. Models with fewer steps will have
    None entries for missing positions.
    """
    # Filter to solutions that have parsed steps
    valid_solutions = [s for s in solutions if s.steps and not s.error]
    if not valid_solutions:
        return []

    # Use model with most steps as reference
    reference = max(valid_solutions, key=lambda s: len(s.steps))
    max_steps = len(reference.steps)

    aligned: list[StepAlignment] = []

    for i in range(max_steps):
        ref_step = reference.steps[i]

        model_steps: dict[str, SolutionStep | None] = {}
        for sol in valid_solutions:
            if i < len(sol.steps):
                model_steps[sol.model_name] = sol.steps[i]
            else:
                model_steps[sol.model_name] = None

        aligned.append(
            StepAlignment(
                canonical_step_number=i + 1,
                description=ref_step.description,
                model_steps=model_steps,
            )
        )

    logger.info(
        "steps_aligned",
        num_steps=len(aligned),
        num_models=len(valid_solutions),
        step_counts={s.model_name: len(s.steps) for s in valid_solutions},
    )

    return aligned
