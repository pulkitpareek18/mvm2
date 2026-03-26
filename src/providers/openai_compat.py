from __future__ import annotations

import structlog
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import RateLimitError, APITimeoutError, APIConnectionError

logger = structlog.get_logger()

# Models that don't support system role — system message gets merged into user message
NO_SYSTEM_ROLE_MODELS = {"google/gemma-2-27b-it", "google/gemma-2-9b-it", "google/gemma-2-2b-it"}

# Models with vision/multimodal support
VISION_MODELS = {"meta-llama/llama-4-scout-17b-16e-instruct"}


class OpenAICompatProvider:
    """Adapter for any OpenAI-compatible API (Groq, NVIDIA NIM, etc.)."""

    def __init__(self, api_key: str, base_url: str, model: str):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_vision(self) -> bool:
        return self._model in VISION_MODELS

    def _prepare_messages(self, messages: list[dict]) -> list[dict]:
        """Prepare messages for the model, handling quirks like no-system-role models."""
        if self._model not in NO_SYSTEM_ROLE_MODELS:
            return messages

        # Merge system message into the first user message
        prepared = []
        system_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                if system_content and msg["role"] == "user":
                    # Merge system into user
                    if isinstance(msg["content"], str):
                        msg = {**msg, "content": f"{system_content}\n\n{msg['content']}"}
                    elif isinstance(msg["content"], list):
                        # Multimodal content — prepend system as text part
                        msg = {**msg, "content": [{"type": "text", "text": system_content}] + msg["content"]}
                    system_content = ""
                prepared.append(msg)

        return prepared if prepared else messages

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIConnectionError)),
        before_sleep=lambda retry_state: structlog.get_logger().warning(
            "retrying_llm_call",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else "unknown",
        ),
    )
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        logger.info("llm_call_start", model=self._model)
        prepared = self._prepare_messages(messages)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=prepared,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        logger.info("llm_call_done", model=self._model, response_len=len(content))
        return content

    async def health_check(self) -> bool:
        try:
            await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=5,
            )
            return True
        except Exception:
            return False
