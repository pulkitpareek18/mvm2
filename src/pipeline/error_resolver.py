"""Error resolution: re-query disagreeing models + multi-agent debate.

OPTIMIZED: Caps total resolution time and LLM calls to prevent runaway debates.
- Max 3 steps get full resolution (requery + debate)
- Max 1 debate round per step
- Steps with 0% agreement (all different) skip debate — not worth it
- Total LLM call budget: 15 calls max for resolution phase
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Awaitable
import asyncio
import re

import structlog

from ..models.problem import MathProblem
from ..models.solution import ModelSolution, SolutionStep
from ..models.verification import StepAlignment
from ..prompts.debate_prompt import REQUERY_PROMPT, DEBATE_PROMPT
from .step_parser import parse_solution
from .consensus import _normalize_answer, _results_match

logger = structlog.get_logger()

EventEmitter = Callable[[dict], Awaitable[None]] | None

# Resolution budget caps
MAX_STEPS_TO_RESOLVE = 3      # Only resolve the top 3 most-contested steps
MAX_DEBATE_ROUNDS = 1          # 1 debate round max per step
MAX_RESOLUTION_LLM_CALLS = 15  # Total LLM call budget for resolution
MIN_AGREEMENT_FOR_DEBATE = 0.2 # Don't debate steps with <20% agreement (everyone disagrees)


async def _requery_model(
    provider,
    problem: MathProblem,
    step: StepAlignment,
    consensus_result: str,
    total_models: int,
) -> SolutionStep | None:
    """Ask a disagreeing model to re-examine its work for a specific step."""
    model_step = step.model_steps.get(provider.model_name)
    if model_step is None:
        return None

    agree_count = sum(
        1 for s in step.model_steps.values()
        if s is not None and _results_match(
            _normalize_answer(s.result or ""), _normalize_answer(consensus_result)
        )
    )

    prompt = REQUERY_PROMPT.format(
        step_number=step.canonical_step_number,
        problem_text=problem.normalized_text or problem.raw_text,
        your_step=f"Description: {model_step.description}\nMATH: {model_step.mathematical_expression}\nRESULT: {model_step.result}",
        agree_count=agree_count,
        total_count=total_models,
        consensus_result=consensus_result,
    )

    messages = [{"role": "user", "content": prompt}]

    try:
        raw = await provider.complete(messages=messages, temperature=0.1, max_tokens=1024)
        temp_solution = ModelSolution(model_name=provider.model_name, raw_response=raw)
        parsed = parse_solution(temp_solution)
        if parsed.steps:
            return parsed.steps[0]
    except Exception as e:
        logger.error("requery_failed", model=provider.model_name, error=str(e))

    return None


async def _run_debate(
    providers: list,
    problem: MathProblem,
    step: StepAlignment,
) -> str | None:
    """Show all models each other's reasoning and ask for the correct answer.
    Only uses 2 fastest providers to save calls."""
    all_solutions_text = ""
    for name, model_step in step.model_steps.items():
        if model_step is not None:
            all_solutions_text += (
                f"--- {name} ---\n"
                f"Description: {model_step.description}\n"
                f"MATH: {model_step.mathematical_expression}\n"
                f"RESULT: {model_step.result}\n\n"
            )

    prompt = DEBATE_PROMPT.format(
        step_number=step.canonical_step_number,
        problem_text=problem.normalized_text or problem.raw_text,
        all_solutions=all_solutions_text,
    )

    messages = [{"role": "user", "content": prompt}]

    # Only ask 2 providers to judge (save calls)
    judge_providers = providers[:2]
    results = await asyncio.gather(
        *[p.complete(messages=messages, temperature=0.1, max_tokens=1024) for p in judge_providers],
        return_exceptions=True,
    )

    result_pattern = re.compile(r"RESULT\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE)
    votes: list[str] = []
    for r in results:
        if isinstance(r, str):
            match = result_pattern.search(r)
            if match:
                votes.append(_normalize_answer(match.group(1).strip()))

    if not votes:
        return None

    counter = Counter(votes)
    return counter.most_common(1)[0][0]


async def resolve_flagged_steps(
    problem: MathProblem,
    solutions: list[ModelSolution],
    aligned_steps: list[StepAlignment],
    providers: list,
    max_debate_rounds: int = MAX_DEBATE_ROUNDS,
    audit_trail: list[str] | None = None,
    emit: EventEmitter = None,
) -> tuple[list[StepAlignment], int]:
    """Resolve disagreements with strict budgets to prevent runaway execution."""
    if audit_trail is None:
        audit_trail = []

    async def _emit(stage: str, event: str, data: dict | None = None, progress: float = 0.0):
        if emit:
            await emit({"stage": stage, "event": event, "data": data or {}, "progress": progress})

    total_debate_rounds = 0
    llm_calls_used = 0
    provider_map = {p.model_name: p for p in providers}

    # Sort flagged steps by agreement ratio (highest first = most likely to resolve)
    flagged = [(i, s) for i, s in enumerate(aligned_steps) if s.flagged]
    flagged.sort(key=lambda x: x[1].agreement_ratio, reverse=True)

    # Cap: only resolve top N steps
    steps_to_resolve = flagged[:MAX_STEPS_TO_RESOLVE]
    skipped = len(flagged) - len(steps_to_resolve)
    if skipped > 0:
        audit_trail.append(f"Skipping {skipped} low-agreement steps (budget: resolve top {MAX_STEPS_TO_RESOLVE})")

    for _, step in steps_to_resolve:
        if llm_calls_used >= MAX_RESOLUTION_LLM_CALLS:
            audit_trail.append(f"LLM call budget exhausted ({llm_calls_used}/{MAX_RESOLUTION_LLM_CALLS})")
            break

        # Skip steps where nobody agrees (0% or very low) — debate won't help
        if step.agreement_ratio < MIN_AGREEMENT_FOR_DEBATE:
            audit_trail.append(f"Step {step.canonical_step_number}: skipped (agreement {step.agreement_ratio:.0%} too low)")
            continue

        logger.info("resolving_step", step=step.canonical_step_number)
        audit_trail.append(f"Step {step.canonical_step_number}: resolving (agreement={step.agreement_ratio:.0%})")

        results = [
            _normalize_answer(s.result or "")
            for s in step.model_steps.values()
            if s is not None and s.result
        ]
        if not results:
            continue

        counter = Counter(results)
        consensus_result = counter.most_common(1)[0][0]

        # Phase 1: Re-query ONLY the disagreeing models (limited to 2 max)
        disagreeing = [
            name for name, s in step.model_steps.items()
            if s is not None and not _results_match(
                _normalize_answer(s.result or ""), consensus_result
            )
        ][:2]  # Max 2 requeries per step

        for name in disagreeing:
            if llm_calls_used >= MAX_RESOLUTION_LLM_CALLS:
                break
            provider = provider_map.get(name)
            if not provider:
                continue

            await _emit("resolution", "requery_sent", {
                "model": name,
                "step_num": step.canonical_step_number,
            }, progress=0.75)

            llm_calls_used += 1
            new_step = await _requery_model(
                provider, problem, step, consensus_result, len(step.model_steps)
            )
            if new_step and _results_match(
                _normalize_answer(new_step.result or ""), consensus_result
            ):
                step.model_steps[name] = new_step
                audit_trail.append(f"  {name.split('/')[-1]} corrected after re-query")

        # Recompute agreement
        present = [s for s in step.model_steps.values() if s is not None]
        if present:
            agree = sum(
                1 for s in present
                if _results_match(_normalize_answer(s.result or ""), consensus_result)
            )
            step.agreement_ratio = agree / len(present)

        if step.agreement_ratio >= 0.6:
            step.flagged = False
            audit_trail.append(f"  Resolved via re-query ({step.agreement_ratio:.0%})")
            continue

        # Phase 2: ONE debate round (uses 2 LLM calls)
        if llm_calls_used + 2 <= MAX_RESOLUTION_LLM_CALLS:
            total_debate_rounds += 1
            audit_trail.append(f"  Debate round 1")

            await _emit("resolution", "debate_round", {
                "round": 1,
                "step_num": step.canonical_step_number,
            }, progress=0.80)

            llm_calls_used += 2  # 2 judges
            debate_answer = await _run_debate(providers, problem, step)
            if debate_answer:
                consensus_result = debate_answer
                agree = sum(
                    1 for s in step.model_steps.values()
                    if s is not None and _results_match(
                        _normalize_answer(s.result or ""), consensus_result
                    )
                )
                present_count = sum(1 for s in step.model_steps.values() if s is not None)
                if present_count > 0:
                    step.agreement_ratio = agree / present_count

                if step.agreement_ratio >= 0.6:
                    step.flagged = False
                    audit_trail.append(f"  Resolved via debate ({step.agreement_ratio:.0%})")

        if step.flagged:
            audit_trail.append(f"  Unresolved")

    remaining = sum(1 for s in aligned_steps if s.flagged)
    await _emit("resolution", "resolution_done", {
        "remaining_flagged": remaining,
        "debate_rounds": total_debate_rounds,
        "llm_calls_used": llm_calls_used,
    }, progress=0.90)

    return aligned_steps, total_debate_rounds
