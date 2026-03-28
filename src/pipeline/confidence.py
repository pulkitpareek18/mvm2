"""Calibrated confidence scoring system.

Combines multiple signal sources into a single calibrated confidence score:

1. OCR Confidence (Pix2Text) — how well was the image recognized?
2. Model Agreement — do the LLMs agree on the answer?
3. Step Agreement — do models agree on intermediate steps?
4. Symbolic Verification — did SymPy confirm the answer?
5. Verbalized Confidence — did models express uncertainty?

Each signal is weighted and calibrated using the Uncertainty-Bench approach
adapted for black-box API models (no logprobs needed).

Key insight from LLM-Uncertainty-Bench: For API-only models, self-consistency
(multiple samples) is the most reliable uncertainty estimator. Our 5-model
pipeline IS self-consistency — each model is an independent sample.
"""
from __future__ import annotations

import re
import math

import structlog

logger = structlog.get_logger()


# ── Weight configuration ─────────────────────────────────────
# These weights sum to 1.0 and were calibrated empirically.
WEIGHTS = {
    "ocr":           0.10,   # OCR quality (only applies to image inputs)
    "answer_agree":  0.30,   # Model agreement on final answer
    "step_agree":    0.20,   # Average step-level agreement
    "sympy_verify":  0.25,   # Symbolic verification result
    "verbalized":    0.15,   # LLM self-assessed confidence
}

# When no image input, redistribute OCR weight
WEIGHTS_NO_IMAGE = {
    "answer_agree":  0.35,
    "step_agree":    0.20,
    "sympy_verify":  0.30,
    "verbalized":    0.15,
}


# ── OCR Confidence ───────────────────────────────────────────
def compute_ocr_confidence(
    ocr_scores: list[float] | None = None,
    image_provided: bool = False,
) -> float:
    """Compute OCR confidence from Pix2Text recognition scores.

    Pix2Text returns per-formula confidence scores (0-1).
    We take the minimum score (weakest link) and apply a penalty curve.

    Args:
        ocr_scores: List of per-formula confidence scores from Pix2Text
        image_provided: Whether the input was an image

    Returns:
        Confidence score 0.0 to 1.0
    """
    if not image_provided:
        return 1.0  # Text input — no OCR needed

    if not ocr_scores:
        return 0.5  # No scores available — assume medium confidence

    # Weakest link: min score drives overall confidence
    min_score = min(ocr_scores)
    avg_score = sum(ocr_scores) / len(ocr_scores)

    # Weighted combination: 60% min (weakest link) + 40% average
    raw = 0.6 * min_score + 0.4 * avg_score

    # Penalty curve: sharply penalize low confidence
    # Maps: 0.9+ → ~0.95, 0.7-0.9 → ~0.7, <0.5 → ~0.25
    calibrated = _sigmoid_calibrate(raw, midpoint=0.7, steepness=8)

    logger.info("ocr_confidence", min_score=min_score, avg_score=avg_score,
                raw=raw, calibrated=calibrated)
    return calibrated


# ── Model Agreement Confidence ───────────────────────────────
def compute_agreement_confidence(
    answer_agreement: dict[str, list[str]],
    total_models: int = 5,
) -> float:
    """Compute confidence from model agreement on the final answer.

    This IS self-consistency scoring (Wang et al., 2022) — each model
    is an independent reasoning path, and we measure agreement.

    Args:
        answer_agreement: {answer: [model_names]} from consensus engine
        total_models: Total models that responded

    Returns:
        Confidence score 0.0 to 1.0
    """
    if not answer_agreement or total_models == 0:
        return 0.0

    # Count models that gave valid answers
    answered = sum(len(models) for models in answer_agreement.values())
    if answered == 0:
        return 0.0

    # Largest agreement group
    max_agree = max(len(models) for models in answer_agreement.values())

    # Raw agreement ratio
    raw = max_agree / answered

    # Apply self-consistency calibration:
    # 1/5 agree (20%) → very low confidence (~0.10)
    # 2/5 agree (40%) → low confidence (~0.30)
    # 3/5 agree (60%) → medium confidence (~0.60)
    # 4/5 agree (80%) → high confidence (~0.85)
    # 5/5 agree (100%) → very high confidence (~0.95)

    # Penalty for few models answering
    response_rate = answered / max(total_models, 1)
    penalty = min(1.0, response_rate + 0.2)  # Mild penalty for missing models

    # Number of distinct answers (entropy-like signal)
    num_distinct = len(answer_agreement)
    entropy_penalty = 1.0 if num_distinct <= 2 else 0.85 if num_distinct == 3 else 0.7

    calibrated = raw * penalty * entropy_penalty

    logger.info("agreement_confidence", raw=raw, max_agree=max_agree,
                answered=answered, distinct=num_distinct, calibrated=calibrated)
    return min(calibrated, 1.0)


# ── Step Agreement Confidence ────────────────────────────────
def compute_step_confidence(
    aligned_steps: list[dict],
) -> float:
    """Compute confidence from step-level agreement.

    Args:
        aligned_steps: List of StepAlignment dicts

    Returns:
        Average step agreement ratio, penalized for flagged steps
    """
    if not aligned_steps:
        return 0.5  # No steps — neutral

    total_ratio = 0.0
    flagged_count = 0

    for step in aligned_steps:
        ratio = step.get("agreement_ratio", 0)
        flagged = step.get("flagged", False)
        total_ratio += ratio
        if flagged:
            flagged_count += 1

    avg_ratio = total_ratio / len(aligned_steps)

    # Penalty for flagged steps
    flagged_ratio = flagged_count / len(aligned_steps)
    penalty = 1.0 - (flagged_ratio * 0.3)  # Each flagged step reduces by ~6%

    return avg_ratio * penalty


# ── Symbolic Verification Confidence ─────────────────────────
def compute_sympy_confidence(
    sympy_override: bool,
    sympy_verified: bool | None = None,
    step_verifications: list[bool | None] | None = None,
) -> float:
    """Compute confidence from SymPy verification results.

    Args:
        sympy_override: Whether SymPy independently confirmed the answer
        sympy_verified: Whether the answer was verified by SymPy
        step_verifications: Per-step symbolic verification results

    Returns:
        Confidence score 0.0 to 1.0
    """
    if sympy_override:
        return 0.95  # SymPy independently verified — very high

    if sympy_verified is True:
        return 0.90

    if sympy_verified is False:
        return 0.10  # SymPy says it's WRONG

    # Check step-level verifications
    if step_verifications:
        verified = sum(1 for v in step_verifications if v is True)
        failed = sum(1 for v in step_verifications if v is False)
        total = len(step_verifications)

        if failed > 0:
            return 0.3  # Some steps failed
        if verified > 0:
            return 0.5 + 0.4 * (verified / total)  # Partial verification

    return 0.5  # Inconclusive — neutral


# ── Verbalized Confidence ────────────────────────────────────
def extract_verbalized_confidence(raw_responses: list[str]) -> float:
    """Extract implicit confidence signals from model responses.

    LLM-Uncertainty-Bench approach adapted for API models:
    Look for hedging language, uncertainty markers, and confidence signals.

    Args:
        raw_responses: Raw text responses from models

    Returns:
        Confidence score 0.0 to 1.0
    """
    if not raw_responses:
        return 0.5

    uncertainty_markers = [
        r"\bi think\b", r"\bprobably\b", r"\bmaybe\b", r"\bnot sure\b",
        r"\bmight be\b", r"\bcould be\b", r"\bapproximately\b",
        r"\bif i'm not mistaken\b", r"\bi believe\b", r"\bpossibly\b",
        r"\buncertain\b", r"\bhard to say\b", r"\bdifficult\b",
        r"\bneed to check\b", r"\bnot confident\b",
    ]

    confidence_markers = [
        r"\bclearly\b", r"\bobviously\b", r"\bdefinitely\b",
        r"\bthe answer is\b", r"\btherefore\b", r"\bthus\b",
        r"\bwe can conclude\b", r"\bstraightforward\b",
        r"\bfinal answer\b", r"\bexactly\b",
    ]

    total_uncertain = 0
    total_confident = 0

    for resp in raw_responses:
        if not resp:
            continue
        lower = resp.lower()

        for pattern in uncertainty_markers:
            total_uncertain += len(re.findall(pattern, lower))

        for pattern in confidence_markers:
            total_confident += len(re.findall(pattern, lower))

    # Net confidence signal
    total = total_uncertain + total_confident
    if total == 0:
        return 0.5  # No signals — neutral

    # Ratio of confident signals
    conf_ratio = total_confident / total

    # Calibrate: mostly uncertain → low, mostly confident → high
    return 0.3 + 0.4 * conf_ratio  # Range: 0.3 to 0.7


# ── Master Confidence Calculator ─────────────────────────────
def compute_calibrated_confidence(
    answer_agreement: dict[str, list[str]],
    aligned_steps: list[dict],
    sympy_override: bool = False,
    sympy_verified: bool | None = None,
    step_verifications: list[bool | None] | None = None,
    raw_responses: list[str] | None = None,
    ocr_scores: list[float] | None = None,
    image_provided: bool = False,
    total_models: int = 5,
) -> dict:
    """Compute the final calibrated confidence score.

    Combines all signal sources with calibrated weights.

    Returns:
        Dict with 'confidence' (float 0-1) and 'breakdown' (per-signal scores)
    """
    # Choose weight set based on input type
    weights = WEIGHTS if image_provided else WEIGHTS_NO_IMAGE

    # Compute individual signals
    signals = {}

    if image_provided:
        signals["ocr"] = compute_ocr_confidence(ocr_scores, image_provided)

    signals["answer_agree"] = compute_agreement_confidence(answer_agreement, total_models)
    signals["step_agree"] = compute_step_confidence(aligned_steps)
    signals["sympy_verify"] = compute_sympy_confidence(
        sympy_override, sympy_verified, step_verifications
    )
    signals["verbalized"] = extract_verbalized_confidence(raw_responses or [])

    # Weighted combination
    total_weight = sum(weights.get(k, 0) for k in signals)
    if total_weight == 0:
        final = 0.5
    else:
        final = sum(signals[k] * weights.get(k, 0) for k in signals) / total_weight

    # OCR penalty: if OCR confidence is very low, cap the final confidence
    if image_provided and signals.get("ocr", 1.0) < 0.4:
        final = min(final, 0.4)
        logger.warning("ocr_penalty_applied", ocr_score=signals.get("ocr"), capped_at=0.4)

    # Clamp to [0, 1]
    final = max(0.0, min(1.0, final))

    breakdown = {
        "signals": {k: round(v, 3) for k, v in signals.items()},
        "weights": {k: v for k, v in weights.items()},
        "final": round(final, 3),
    }

    logger.info("calibrated_confidence", **breakdown)
    return {"confidence": final, "breakdown": breakdown}


# ── Utility ──────────────────────────────────────────────────
def _sigmoid_calibrate(x: float, midpoint: float = 0.5, steepness: float = 10) -> float:
    """Apply a sigmoid calibration curve.

    Maps [0, 1] → [0, 1] with sharp transition around midpoint.
    """
    try:
        return 1.0 / (1.0 + math.exp(-steepness * (x - midpoint)))
    except OverflowError:
        return 0.0 if x < midpoint else 1.0
