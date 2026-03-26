from __future__ import annotations

import base64

import structlog

from ..models.problem import MathProblem, ProblemType

logger = structlog.get_logger()

CLASSIFICATION_PROMPT = """Classify this math problem into exactly ONE category. Respond with ONLY the category name, nothing else.

Categories:
- arithmetic (basic operations: addition, subtraction, multiplication, division, percentages)
- algebra (equations, variables, expressions, factoring, polynomials, inequalities)
- geometry (shapes, areas, perimeters, volumes, angles, triangles, coordinate geometry)
- word_problem (real-world scenarios requiring math to solve)
- number_theory (primes, factors, divisibility, GCD, LCM, modular arithmetic)
- statistics (mean, median, mode, standard deviation, data analysis)
- calculus (derivatives, integrals, limits, series, differential calculus, integral calculus)
- linear_algebra (matrices, determinants, eigenvalues, vector spaces, linear transformations)
- differential_equations (ODEs, PDEs, initial value problems, boundary value problems)
- trigonometry (trig functions, identities, equations, inverse trig, unit circle)
- probability (distributions, expected value, Bayes theorem, combinatorial probability)
- discrete_math (combinatorics, graph theory, logic, set theory, recurrence relations)
- competition (olympiad problems, AMC, AIME, IMO style, contest math)

Problem: {problem_text}

Category:"""

OCR_PROMPT = """Extract the math problem from this image. Return ONLY the problem text, using standard mathematical notation. Use ^ for exponents, * for multiplication, / for division. If there are multiple problems, extract only the first one."""


async def extract_text_from_image(image_data: bytes, vision_provider) -> str:
    """Use a vision-capable LLM to extract math text from an image."""
    b64 = base64.b64encode(image_data).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
            ],
        }
    ]

    try:
        text = await vision_provider.complete(
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        logger.info("ocr_complete", text_length=len(text))
        return text.strip()
    except Exception as e:
        logger.error("ocr_failed", error=str(e))
        return ""


async def classify_problem(problem_text: str, provider) -> ProblemType:
    """Classify a math problem into a category using a lightweight LLM call."""
    messages = [
        {
            "role": "user",
            "content": CLASSIFICATION_PROMPT.format(problem_text=problem_text),
        }
    ]

    try:
        response = await provider.complete(
            messages=messages,
            temperature=0.0,
            max_tokens=20,
        )
        category = response.strip().lower().replace(" ", "_")

        try:
            return ProblemType(category)
        except ValueError:
            logger.warning("unknown_problem_type", response=category)
            return ProblemType.UNKNOWN
    except Exception as e:
        logger.warning("classification_failed", error=str(e))
        return ProblemType.UNKNOWN


async def process_input(
    problem: MathProblem,
    providers: list,
) -> MathProblem:
    """Process and normalize input: OCR if image, classify problem type."""

    # Find a vision-capable provider for OCR
    vision_provider = None
    cheapest_provider = None

    for p in providers:
        if p.supports_vision:
            vision_provider = p
        if cheapest_provider is None:
            cheapest_provider = p  # first provider is assumed cheapest

    # OCR: extract text from image if provided
    if problem.image_data:
        if vision_provider:
            logger.info("ocr_starting", provider=vision_provider.model_name)
            extracted_text = await extract_text_from_image(
                problem.image_data, vision_provider
            )
            if extracted_text:
                problem.normalized_text = extracted_text
                if not problem.raw_text or problem.raw_text.strip() == "":
                    problem.raw_text = extracted_text
                logger.info("ocr_success", extracted_text=extracted_text[:100])
            else:
                logger.error("ocr_returned_empty")
        else:
            logger.error("no_vision_provider", available=[p.model_name for p in providers])

    if not problem.normalized_text:
        problem.normalized_text = problem.raw_text

    # Classify the problem
    if cheapest_provider:
        problem.problem_type = await classify_problem(
            problem.normalized_text or problem.raw_text,
            cheapest_provider,
        )
        logger.info("problem_classified", type=problem.problem_type)

    return problem
