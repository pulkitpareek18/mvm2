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


async def _try_pix2text_ocr(image_data: bytes) -> tuple[str, list[float]]:
    """Try OCR with Pix2Text. Returns (extracted_text, confidence_scores).

    Pix2Text is specialized for math OCR and provides per-formula
    confidence scores that we use for calibrated confidence scoring.
    """
    import asyncio
    from io import BytesIO
    from PIL import Image

    def _run_pix2text(img_bytes: bytes) -> tuple[str, list[float]]:
        from pix2text import Pix2Text

        p2t = Pix2Text.from_config()
        img = Image.open(BytesIO(img_bytes))

        # recognize_formula returns detailed dict with scores when return_text=False
        result = p2t.recognize(img, file_type='text_formula')

        # Extract text and scores
        if isinstance(result, str):
            return result, [0.8]  # String result — no detailed scores
        elif isinstance(result, list):
            texts = []
            scores = []
            for item in result:
                if isinstance(item, dict):
                    texts.append(item.get('text', ''))
                    scores.append(item.get('score', 0.8))
                elif isinstance(item, str):
                    texts.append(item)
                    scores.append(0.8)
            return ' '.join(texts), scores if scores else [0.8]
        else:
            return str(result), [0.8]

    # Run in thread pool to avoid blocking async loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_pix2text, image_data)


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
    # Strategy: Try Pix2Text first (better math OCR + confidence scores),
    # fall back to LLM vision if Pix2Text unavailable or fails.
    if problem.image_data:
        ocr_done = False

        # Tier 1: Pix2Text (specialized math OCR with confidence scores)
        try:
            extracted, ocr_scores = await _try_pix2text_ocr(problem.image_data)
            if extracted:
                problem.normalized_text = extracted
                if not problem.raw_text or problem.raw_text.strip() == "":
                    problem.raw_text = extracted
                # Store OCR scores for confidence calibration
                problem._ocr_scores = ocr_scores  # type: ignore
                logger.info("pix2text_ocr_success", text=extracted[:100],
                           scores=ocr_scores)
                ocr_done = True
        except Exception as e:
            logger.warning("pix2text_ocr_failed", error=str(e))

        # Tier 2: LLM Vision fallback
        if not ocr_done and vision_provider:
            logger.info("ocr_fallback_llm", provider=vision_provider.model_name)
            extracted_text = await extract_text_from_image(
                problem.image_data, vision_provider
            )
            if extracted_text:
                problem.normalized_text = extracted_text
                if not problem.raw_text or problem.raw_text.strip() == "":
                    problem.raw_text = extracted_text
                logger.info("llm_ocr_success", extracted_text=extracted_text[:100])
            else:
                logger.error("ocr_returned_empty")
        elif not ocr_done:
            logger.error("no_ocr_available", has_vision=bool(vision_provider))

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
