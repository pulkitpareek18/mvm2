"""Re-parse module: when a model's output fails structured parsing,
use a fast model to reformat it into the STEP/MATH/RESULT format."""

from __future__ import annotations

import structlog

from ..models.solution import ModelSolution
from .step_parser import parse_solution

logger = structlog.get_logger()

REPARSE_PROMPT = """The following is a math solution from an AI model, but it's not in the correct structured format.
Please reformat it into this EXACT format — do NOT change any math, just reformat:

STEP 1: [description]
MATH: [expression]
RESULT: [result]

STEP 2: [description]
MATH: [expression]
RESULT: [result]

... (continue for all steps)

FINAL ANSWER: [answer]

RULES:
- Extract EVERY step the model performed
- Keep all mathematical expressions exactly as they are
- If a step has no computation, use MATH: N/A and RESULT: N/A
- The FINAL ANSWER must be ONLY the final value/expression
- Do NOT add any text before STEP 1 or after FINAL ANSWER

Here is the solution to reformat:

{raw_solution}"""


async def reparse_failed_solutions(
    solutions: list[ModelSolution],
    providers: list,
    emit=None,
) -> list[ModelSolution]:
    """For solutions that failed parsing, try re-parsing with a fast model."""

    # Find a fast provider for re-parsing (first in list = cheapest)
    reparse_provider = providers[0] if providers else None
    if not reparse_provider:
        return solutions

    reparsed = []
    for sol in solutions:
        # Only re-parse if: model responded OK but we got 0 steps
        if sol.error is None and len(sol.steps) == 0 and sol.raw_response:
            logger.info("reparse_attempt", model=sol.model_name)

            if emit:
                await emit({
                    "stage": "parsing", "event": "reparse_start",
                    "data": {"model": sol.model_name}, "progress": 0.42,
                })

            try:
                # Truncate raw response to avoid token limits
                raw_truncated = sol.raw_response[:6000]
                messages = [
                    {"role": "user", "content": REPARSE_PROMPT.format(raw_solution=raw_truncated)}
                ]

                reformatted = await reparse_provider.complete(
                    messages=messages,
                    temperature=0.0,
                    max_tokens=4096,
                )

                # Try parsing the reformatted output
                temp = ModelSolution(
                    model_name=sol.model_name,
                    raw_response=reformatted,
                    latency_ms=sol.latency_ms,
                )
                parsed = parse_solution(temp)

                if parsed.steps:
                    logger.info("reparse_success", model=sol.model_name, steps=len(parsed.steps))
                    # Keep original raw_response but use new parsed steps
                    sol.steps = parsed.steps
                    sol.final_answer = parsed.final_answer or sol.final_answer
                    sol.error = None

                    if emit:
                        await emit({
                            "stage": "parsing", "event": "reparse_done",
                            "data": {"model": sol.model_name, "steps": len(parsed.steps), "status": "ok"},
                            "progress": 0.44,
                        })
                else:
                    logger.warning("reparse_still_failed", model=sol.model_name)
                    if emit:
                        await emit({
                            "stage": "parsing", "event": "reparse_done",
                            "data": {"model": sol.model_name, "steps": 0, "status": "failed"},
                            "progress": 0.44,
                        })

            except Exception as e:
                logger.error("reparse_error", model=sol.model_name, error=str(e))

        reparsed.append(sol)

    return reparsed
