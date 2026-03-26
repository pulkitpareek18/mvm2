from __future__ import annotations

import asyncio
import time
from typing import Callable, Awaitable

import structlog

from ..models.problem import MathProblem
from ..models.solution import ModelSolution
from ..prompts.solver_prompt import SOLVER_SYSTEM_PROMPT, SOLVER_USER_PROMPT

logger = structlog.get_logger()

# Type alias for the event emitter callback
EventEmitter = Callable[[dict], Awaitable[None]] | None


async def _solve_single(
    provider,
    problem: MathProblem,
    temperature: float,
    max_tokens: int,
) -> ModelSolution:
    """Send a problem to a single model and return its solution."""
    problem_text = problem.normalized_text or problem.raw_text

    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": SOLVER_USER_PROMPT.format(problem_text=problem_text)},
    ]

    start = time.monotonic()
    raw_response = await provider.complete(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency_ms = (time.monotonic() - start) * 1000

    return ModelSolution(
        model_name=provider.model_name,
        raw_response=raw_response,
        latency_ms=latency_ms,
    )


async def solve_with_all_models(
    problem: MathProblem,
    providers: list,
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout: float = 60.0,
    emit: EventEmitter = None,
) -> list[ModelSolution]:
    """Dispatch problem to all models in parallel, emit events as each model completes."""

    async def _emit(stage: str, event: str, data: dict | None = None, progress: float = 0.0):
        if emit:
            await emit({"stage": stage, "event": event, "data": data or {}, "progress": progress})

    # Emit model_start for each model
    for p in providers:
        await _emit("solving", "model_start", {"model": p.model_name})

    # Create tasks with model name tracking
    async def _tracked_solve(provider):
        try:
            result = await asyncio.wait_for(
                _solve_single(provider, problem, temperature, max_tokens),
                timeout=timeout,
            )
            return result
        except Exception as e:
            return ModelSolution(model_name=provider.model_name, error=str(e))

    # Use as_completed to emit per-model events as they finish
    tasks = {asyncio.ensure_future(_tracked_solve(p)): p for p in providers}
    solutions: list[ModelSolution] = []
    completed = 0

    for coro in asyncio.as_completed(tasks.keys()):
        result = await coro
        completed += 1
        solutions.append(result)

        status = "ok" if result.error is None else "error"
        progress = (completed / len(providers)) * 0.4  # solving is 0-40% of pipeline

        await _emit("solving", "model_done", {
            "model": result.model_name,
            "latency_ms": round(result.latency_ms, 1),
            "status": status,
            "error": result.error,
            "completed": completed,
            "total": len(providers),
        }, progress=progress)

        logger.info(
            "model_completed",
            model=result.model_name,
            status=status,
            latency_ms=round(result.latency_ms, 1),
        )

    successful = sum(1 for s in solutions if s.error is None)
    logger.info("parallel_solve_done", total=len(solutions), successful=successful)
    return solutions
