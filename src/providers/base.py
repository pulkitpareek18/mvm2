from __future__ import annotations

from typing import Protocol


class LLMProvider(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def supports_vision(self) -> bool: ...

    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str: ...
