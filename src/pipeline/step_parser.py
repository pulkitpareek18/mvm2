from __future__ import annotations

import re

import structlog

from ..models.solution import ModelSolution, SolutionStep

logger = structlog.get_logger()

# ── Think tag handling ──────────────────────────────────────
# Qwen3/DeepSeek-R1 wrap reasoning in <think>...</think>.
# The structured answer may be INSIDE or AFTER the think block.
# Strategy: try parsing the full text first, then try after stripping.
THINK_TAG_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking_tags(text: str) -> str:
    """Strip <think>...</think> tags."""
    return THINK_TAG_PATTERN.sub("", text).strip()


def _extract_after_think(text: str) -> str:
    """Extract content AFTER the last </think> tag."""
    idx = text.rfind("</think>")
    if idx != -1:
        return text[idx + 8:].strip()
    return text


# ── Step extraction patterns ────────────────────────────────
# Primary: strict STEP/MATH/RESULT format
STEP_STRICT = re.compile(
    r"STEP\s+(\d+)\s*:\s*(.+?)(?:\n|\r\n?)"
    r"MATH\s*:\s*(.+?)(?:\n|\r\n?)"
    r"RESULT\s*:\s*(.+?)(?:\n|\r\n?|$)",
    re.IGNORECASE | re.DOTALL,
)

# Fallback: just STEP N: description with any content until next STEP or FINAL
STEP_LOOSE = re.compile(
    r"STEP\s+(\d+)\s*[:\-\.]\s*(.+?)(?=STEP\s+\d+|FINAL\s+ANSWER|$)",
    re.IGNORECASE | re.DOTALL,
)

# Extract MATH: and RESULT: within a loose step block
MATH_LINE = re.compile(r"MATH\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
RESULT_LINE = re.compile(r"RESULT\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)

FINAL_ANSWER_PATTERN = re.compile(
    r"FINAL\s+ANSWER\s*:\s*(.+?)(?:\n|\r\n?|$)",
    re.IGNORECASE,
)

# Also catch boxed answers common in math: \boxed{...}
BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}")


def _parse_steps_strict(text: str) -> list[SolutionStep]:
    """Try strict STEP/MATH/RESULT parsing."""
    steps = []
    for match in STEP_STRICT.finditer(text):
        steps.append(SolutionStep(
            step_number=int(match.group(1)),
            description=match.group(2).strip(),
            mathematical_expression=match.group(3).strip() if match.group(3).strip().upper() != "N/A" else None,
            result=match.group(4).strip() if match.group(4).strip().upper() != "N/A" else None,
        ))
    return steps


def _parse_steps_loose(text: str) -> list[SolutionStep]:
    """Fallback: looser parsing that finds STEP blocks and extracts MATH/RESULT within them."""
    steps = []
    for match in STEP_LOOSE.finditer(text):
        step_num = int(match.group(1))
        block = match.group(2).strip()

        # First line is the description
        lines = block.split("\n")
        description = lines[0].strip() if lines else ""

        # Search for MATH and RESULT within the block
        math_match = MATH_LINE.search(block)
        result_match = RESULT_LINE.search(block)

        math_expr = math_match.group(1).strip() if math_match else None
        result_val = result_match.group(1).strip() if result_match else None

        if math_expr and math_expr.upper() == "N/A":
            math_expr = None
        if result_val and result_val.upper() == "N/A":
            result_val = None

        steps.append(SolutionStep(
            step_number=step_num,
            description=description,
            mathematical_expression=math_expr,
            result=result_val,
        ))
    return steps


def _extract_final_answer(text: str) -> str:
    """Extract final answer from text."""
    # Try FINAL ANSWER: pattern
    match = FINAL_ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).strip()

    # Try \boxed{} pattern
    boxed = BOXED_PATTERN.findall(text)
    if boxed:
        return boxed[-1].strip()

    return ""


def parse_solution(solution: ModelSolution) -> ModelSolution:
    """Parse raw LLM response into structured SolutionStep objects and final answer.

    Handles:
    - Standard STEP/MATH/RESULT format
    - Qwen3/DeepSeek <think> tags (tries content after tags, then inside)
    - Loose STEP format without strict MATH/RESULT lines
    - LaTeX \boxed{} answers
    """
    if solution.error:
        return solution

    raw = solution.raw_response.strip()
    if not raw:
        solution.error = "empty_response"
        return solution

    # Try multiple parsing strategies in order of preference
    steps: list[SolutionStep] = []
    final_answer = ""

    # Strategy 1: Parse the raw text directly (works for non-thinking models)
    steps = _parse_steps_strict(raw)
    final_answer = _extract_final_answer(raw)

    # Strategy 2: If no steps found and has <think> tags, try content AFTER </think>
    if not steps and "<think>" in raw:
        after_think = _extract_after_think(raw)
        if after_think:
            steps = _parse_steps_strict(after_think)
            if not final_answer:
                final_answer = _extract_final_answer(after_think)

    # Strategy 3: If still no steps, try loose parsing on full text (stripped of think tags)
    if not steps:
        cleaned = _strip_thinking_tags(raw) if "<think>" in raw else raw
        steps = _parse_steps_loose(cleaned)
        if not final_answer:
            final_answer = _extract_final_answer(cleaned)

    # Strategy 4: If still no steps, try loose parsing on content inside think tags
    if not steps and "<think>" in raw:
        # Sometimes the model puts structured content inside think tags
        think_content = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_content:
            inner = think_content.group(1)
            steps = _parse_steps_loose(inner)
            if not final_answer:
                final_answer = _extract_final_answer(inner)

    # Last resort: grab the last non-empty line as the final answer
    if not steps and not final_answer:
        lines = [l.strip() for l in raw.split("\n") if l.strip()]
        # Filter out think tags
        lines = [l for l in lines if not l.startswith("<think>") and not l.startswith("</think>")]
        if lines:
            final_answer = lines[-1][:200]
        solution.error = "parse_failed"
        logger.warning("step_parse_failed", model=solution.model_name, response_len=len(raw))

    if not steps and final_answer:
        logger.warning("no_structured_steps", model=solution.model_name, has_answer=True)

    solution.steps = steps
    solution.final_answer = final_answer

    logger.info(
        "parse_result",
        model=solution.model_name,
        steps=len(steps),
        has_answer=bool(final_answer),
    )

    return solution


def parse_all_solutions(solutions: list[ModelSolution]) -> list[ModelSolution]:
    """Parse all model solutions."""
    return [parse_solution(s) for s in solutions]
