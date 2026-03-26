from __future__ import annotations

from typing import Callable, Awaitable

import structlog

from ..config import settings
from ..models.problem import MathProblem
from ..models.verification import VerificationResult
from .solver import solve_with_all_models
from .step_parser import parse_all_solutions
from .step_aligner import align_steps
from .consensus import (
    compute_step_consensus,
    compute_answer_agreement,
    get_consensus_answer,
)
from .error_resolver import resolve_flagged_steps

logger = structlog.get_logger()

EventEmitter = Callable[[dict], Awaitable[None]] | None


async def verify_math_problem(
    problem: MathProblem,
    providers: list,
    emit: EventEmitter = None,
) -> VerificationResult:
    """Top-level pipeline with real-time event streaming."""

    async def _emit(stage: str, event: str, data: dict | None = None, progress: float = 0.0):
        if emit:
            await emit({"stage": stage, "event": event, "data": data or {}, "progress": progress})

    audit_trail: list[str] = []
    audit_trail.append(f"Problem: {problem.raw_text[:100]}")
    audit_trail.append(f"Models: {[p.model_name for p in providers]}")

    # === STAGE 0: Input processing (OCR + classification) ===
    await _emit("init", "pipeline_start", {
        "problem": problem.raw_text[:200],
        "models": [p.model_name for p in providers],
    }, progress=0.0)

    from .input_processor import process_input
    problem = await process_input(problem, providers)
    if problem.normalized_text and problem.normalized_text != problem.raw_text:
        audit_trail.append(f"OCR/normalized: {problem.normalized_text[:100]}")
    if problem.problem_type.value != "unknown":
        audit_trail.append(f"Classified as: {problem.problem_type.value}")

    # === STAGE 1: Solve with all models ===
    solutions = await solve_with_all_models(
        problem=problem,
        providers=providers,
        temperature=settings.solver_temperature,
        max_tokens=settings.max_solver_tokens,
        timeout=settings.parallel_timeout_seconds,
        emit=emit,  # pass through for per-model events
    )

    successful = [s for s in solutions if s.error is None]
    audit_trail.append(f"Solver: {len(successful)}/{len(solutions)} models responded successfully")

    if len(successful) < settings.min_models_required:
        await _emit("done", "error", {
            "message": f"Only {len(successful)} models responded, need {settings.min_models_required}",
        }, progress=1.0)
        return VerificationResult(
            problem_id=problem.id,
            problem_text=problem.raw_text,
            final_answer="ERROR: insufficient model responses",
            confidence=0.0,
            audit_trail=audit_trail + [
                f"FAILED: Only {len(successful)} models responded, need {settings.min_models_required}"
            ],
        )

    # === STAGE 3: Parse responses ===
    solutions = parse_all_solutions(solutions)
    parsed_ok = sum(1 for s in solutions if s.steps and not s.error)
    parse_failed = sum(1 for s in solutions if s.error is None and not s.steps)
    audit_trail.append(f"Parser: {parsed_ok}/{len(solutions)} solutions parsed successfully")

    await _emit("parsing", "parse_done", {
        "parsed_count": parsed_ok,
        "failed_count": len(solutions) - parsed_ok,
    }, progress=0.42)

    # === STAGE 3b: Re-parse failed solutions ===
    if parse_failed > 0:
        audit_trail.append(f"Re-parsing {parse_failed} failed solutions...")
        from .reparser import reparse_failed_solutions
        solutions = await reparse_failed_solutions(solutions, providers, emit=emit)
        reparsed_ok = sum(1 for s in solutions if s.steps and not s.error)
        if reparsed_ok > parsed_ok:
            audit_trail.append(f"Re-parse recovered {reparsed_ok - parsed_ok} additional solutions")
        parsed_ok = reparsed_ok

    await _emit("parsing", "parse_complete", {
        "parsed_count": parsed_ok,
        "total": len(solutions),
    }, progress=0.45)

    # === STAGE 4: Answer agreement ===
    answer_agreement = compute_answer_agreement(solutions)
    consensus_answer, answer_confidence = get_consensus_answer(answer_agreement)

    audit_trail.append(f"Answer agreement: {answer_agreement}")
    audit_trail.append(f"Consensus answer: {consensus_answer} (confidence: {answer_confidence:.0%})")

    await _emit("consensus", "answer_agreement", {
        "agreement": answer_agreement,
        "consensus_answer": consensus_answer,
        "confidence": round(answer_confidence, 3),
    }, progress=0.50)

    # === STAGE 5: Align steps ===
    aligned_steps = align_steps(solutions)
    audit_trail.append(f"Aligned {len(aligned_steps)} canonical steps")

    await _emit("alignment", "steps_aligned", {
        "step_count": len(aligned_steps),
    }, progress=0.55)

    # === STAGE 6: Per-step consensus ===
    aligned_steps = compute_step_consensus(aligned_steps, settings.consensus_threshold)
    flagged_count = sum(1 for s in aligned_steps if s.flagged)

    if flagged_count > 0:
        audit_trail.append(f"{flagged_count} steps flagged for disagreement")

    await _emit("consensus", "step_consensus", {
        "flagged_count": flagged_count,
        "total_steps": len(aligned_steps),
    }, progress=0.60)

    # === STAGE 7: Symbolic verification ===
    symbolic_override = False
    if settings.enable_symbolic_verification:
        await _emit("symbolic", "symbolic_start", {}, progress=0.62)

        try:
            from ..symbolic.verifier import verify_all_steps

            aligned_steps, sym_results = verify_all_steps(aligned_steps)
            verified = sum(1 for r in sym_results if r is True)
            failed = sum(1 for r in sym_results if r is False)
            inconclusive = sum(1 for r in sym_results if r is None)
            audit_trail.append(
                f"Symbolic verification: {verified} verified, {failed} failed, {inconclusive} inconclusive"
            )

            # Emit per-step symbolic results
            for step, sym_result in zip(aligned_steps, sym_results):
                await _emit("symbolic", "step_verified", {
                    "step_num": step.canonical_step_number,
                    "result": sym_result,  # True/False/None
                }, progress=0.62 + (0.08 * step.canonical_step_number / max(len(aligned_steps), 1)))

                if sym_result is False and not step.flagged:
                    step.flagged = True
                    flagged_count += 1
                    audit_trail.append(
                        f"Step {step.canonical_step_number}: flagged by symbolic verification"
                    )
        except Exception as e:
            audit_trail.append(f"Symbolic verification skipped: {e}")

        await _emit("symbolic", "symbolic_done", {
            "verified": verified if settings.enable_symbolic_verification else 0,
            "failed": failed if settings.enable_symbolic_verification else 0,
            "inconclusive": inconclusive if settings.enable_symbolic_verification else 0,
        }, progress=0.70)

    # === STAGE 8: Error resolution ===
    debate_rounds = 0
    if flagged_count > 0:
        audit_trail.append("Starting error resolution...")
        await _emit("resolution", "resolution_start", {
            "flagged_count": flagged_count,
        }, progress=0.72)

        aligned_steps, debate_rounds = await resolve_flagged_steps(
            problem=problem,
            solutions=solutions,
            aligned_steps=aligned_steps,
            providers=providers,
            max_debate_rounds=settings.max_debate_rounds,
            audit_trail=audit_trail,
            emit=emit,
        )

        remaining_flagged = sum(1 for s in aligned_steps if s.flagged)
        audit_trail.append(f"After resolution: {remaining_flagged} steps still flagged")

    # === STAGE 9: Independent Answer Verification (SymPy) ===
    await _emit("verification", "verify_start", {}, progress=0.92)

    try:
        from ..symbolic.answer_verifier import verify_answers

        verification = verify_answers(
            problem_text=problem.normalized_text or problem.raw_text,
            answer_agreement=answer_agreement,
        )

        if verification["sympy_computed"]:
            audit_trail.append(f"SymPy independent answer: {verification['sympy_answer']} (method: {verification['method']})")
            audit_trail.append(f"SymPy numerical: {verification['sympy_numerical']}")

            for ans, verified in verification["verified_answers"].items():
                status = "CORRECT" if verified else "WRONG" if verified is False else "unverifiable"
                audit_trail.append(f"  Model answer '{ans[:40]}': {status}")

            if verification["best_answer"]:
                # Override consensus with verified answer
                old_answer = consensus_answer
                consensus_answer = verification["best_answer"]
                symbolic_override = True

                if verification["best_answer_models"] == ["SymPy (independent computation)"]:
                    audit_trail.append(f"OVERRIDE: No model matched. Using SymPy answer: {consensus_answer}")
                    answer_confidence = 0.7  # SymPy is reliable but not always right for complex problems
                else:
                    audit_trail.append(f"VERIFIED: Answer '{consensus_answer}' confirmed by SymPy")
                    answer_confidence = 0.95  # High confidence — model + SymPy agree

                await _emit("verification", "verify_done", {
                    "sympy_computed": True,
                    "verified_answer": str(consensus_answer),
                    "method": verification["method"],
                    "override": old_answer != consensus_answer,
                }, progress=0.95)
            else:
                audit_trail.append("SymPy could not verify any answer")
                await _emit("verification", "verify_done", {
                    "sympy_computed": True, "verified_answer": None,
                }, progress=0.95)
        else:
            audit_trail.append("SymPy could not independently compute this problem")
            await _emit("verification", "verify_done", {
                "sympy_computed": False,
            }, progress=0.95)

    except Exception as e:
        audit_trail.append(f"Answer verification error: {e}")
        logger.warning("answer_verification_error", error=str(e))

    # === STAGE 10: Final confidence ===
    if aligned_steps:
        avg_agreement = sum(s.agreement_ratio for s in aligned_steps) / len(aligned_steps)
        final_confidence = (answer_confidence * 0.6) + (avg_agreement * 0.4)
    else:
        final_confidence = answer_confidence

    # Boost confidence based on verification strength
    if symbolic_override:
        # SymPy independently verified the answer
        if answer_confidence >= 0.9:
            # Models agree + SymPy confirms = very high confidence
            final_confidence = max(final_confidence, 0.95)
        else:
            # SymPy override (models disagreed but SymPy knows) = good but not perfect
            final_confidence = max(final_confidence, 0.80)

    audit_trail.append(f"Final confidence: {final_confidence:.0%}")

    # Build result
    model_solutions_dict = {}
    for sol in solutions:
        model_solutions_dict[sol.model_name] = [
            step.model_dump() for step in sol.steps
        ]

    result = VerificationResult(
        problem_id=problem.id,
        problem_text=problem.raw_text,
        model_solutions=model_solutions_dict,
        aligned_steps=aligned_steps,
        final_answer=consensus_answer,
        confidence=final_confidence,
        answer_agreement=answer_agreement,
        debate_rounds=debate_rounds,
        symbolic_override=symbolic_override,
        audit_trail=audit_trail,
    )

    await _emit("done", "result", {}, progress=1.0)

    return result
