from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..models import MathProblem
from ..models.verification import VerificationResult
from ..providers.registry import build_providers

router = APIRouter()

# Lazy-init providers on first request
_providers = None


def get_providers():
    global _providers
    if _providers is None:
        _providers = build_providers()
    return _providers


class SolveRequest(BaseModel):
    text: str
    image_base64: str | None = None


class HealthStatus(BaseModel):
    status: str
    providers: dict[str, bool]


@router.get("/health", response_model=HealthStatus)
async def health():
    """Health check — pings each LLM provider."""
    providers = get_providers()
    statuses = {}
    for provider in providers:
        try:
            ok = await provider.health_check()
            statuses[provider.model_name] = ok
        except Exception:
            statuses[provider.model_name] = False
    overall = "ok" if any(statuses.values()) else "degraded"
    return HealthStatus(status=overall, providers=statuses)


@router.post("/solve", response_model=VerificationResult)
async def solve(request: SolveRequest):
    """Accept a math problem, run full verification pipeline, return verified result."""
    providers = get_providers()
    if len(providers) < 2:
        raise HTTPException(status_code=503, detail="Need at least 2 LLM providers configured")

    image_data = None
    if request.image_base64:
        import base64
        image_data = base64.b64decode(request.image_base64)

    problem = MathProblem(raw_text=request.text, image_data=image_data)

    from ..pipeline.orchestrator import verify_math_problem
    result = await verify_math_problem(problem, providers)
    return result


@router.post("/solve/stream")
async def solve_stream(request: SolveRequest):
    """SSE streaming endpoint — emits real-time events as the pipeline executes.

    Each event is a JSON line:
    data: {"stage": "solving", "event": "model_done", "data": {...}, "progress": 0.15}
    """
    providers = get_providers()
    if len(providers) < 2:
        raise HTTPException(status_code=503, detail="Need at least 2 LLM providers configured")

    image_data = None
    if request.image_base64:
        import base64
        image_data = base64.b64decode(request.image_base64)

    problem = MathProblem(raw_text=request.text, image_data=image_data)

    async def event_generator():
        # Queue for events from the pipeline
        queue: asyncio.Queue[dict | None] = asyncio.Queue()

        async def emit(event_data: dict):
            await queue.put(event_data)

        async def run_pipeline():
            try:
                from ..pipeline.orchestrator import verify_math_problem
                result = await verify_math_problem(problem, providers, emit=emit)
                # Send the final result as the last event
                await queue.put({
                    "stage": "done",
                    "event": "final_result",
                    "data": result.model_dump(),
                    "progress": 1.0,
                })
            except Exception as e:
                await queue.put({
                    "stage": "error",
                    "event": "pipeline_error",
                    "data": {"error": str(e)},
                    "progress": 1.0,
                })
            finally:
                await queue.put(None)  # sentinel to end the stream

        # Start pipeline in background
        task = asyncio.create_task(run_pipeline())

        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield f"data: {json.dumps(event, default=str)}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/report/pdf")
async def generate_pdf(request: Request):
    """Generate a PDF verification report from result data."""
    import io
    from .pdf_report import generate_pdf_report

    try:
        data = await request.json()
        pdf_bytes = generate_pdf_report(data)
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=mvm2_report.pdf"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")
