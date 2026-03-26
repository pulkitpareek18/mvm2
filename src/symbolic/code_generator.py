from __future__ import annotations

import structlog

from ..prompts.code_gen_prompt import CODE_GEN_SYSTEM_PROMPT, CODE_GEN_USER_PROMPT
from ..models.solution import SolutionStep

logger = structlog.get_logger()


async def generate_verification_code(
    step: SolutionStep,
    provider,
) -> str | None:
    """Use an LLM to convert a math step into Python/SymPy verification code.

    Returns the generated code string, or None if generation fails.
    """
    if not step.mathematical_expression or step.mathematical_expression == "N/A":
        return None

    messages = [
        {"role": "system", "content": CODE_GEN_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": CODE_GEN_USER_PROMPT.format(
                description=step.description,
                expression=step.mathematical_expression,
                result=step.result or "N/A",
            ),
        },
    ]

    try:
        code = await provider.complete(
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )

        # Strip markdown code fences if present
        code = code.strip()
        if code.startswith("```python"):
            code = code[len("```python") :].strip()
        elif code.startswith("```"):
            code = code[3:].strip()
        if code.endswith("```"):
            code = code[:-3].strip()

        return code
    except Exception as e:
        logger.warning("code_generation_failed", error=str(e))
        return None
