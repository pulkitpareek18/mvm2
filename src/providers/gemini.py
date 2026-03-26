from __future__ import annotations

import structlog
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class GeminiProvider:
    """Adapter for Google Gemini API."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._client = genai.Client(api_key=api_key)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_vision(self) -> bool:
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
    )
    async def complete(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        logger.info("gemini_call_start", model=self._model)

        # Convert OpenAI-style messages to Gemini format
        contents = []
        system_instruction = None

        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                parts = []
                if isinstance(msg["content"], str):
                    parts.append(types.Part.from_text(text=msg["content"]))
                elif isinstance(msg["content"], list):
                    # Multimodal: list of content parts (text + images)
                    for part in msg["content"]:
                        if part.get("type") == "text":
                            parts.append(types.Part.from_text(text=part["text"]))
                        elif part.get("type") == "image_url":
                            # Handle base64 image data
                            url = part["image_url"]["url"]
                            if url.startswith("data:"):
                                import base64

                                # Parse data URI: data:image/png;base64,<data>
                                header, b64data = url.split(",", 1)
                                mime_type = header.split(":")[1].split(";")[0]
                                image_bytes = base64.b64decode(b64data)
                                parts.append(
                                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                                )
                contents.append(types.Content(role="user", parts=parts))
            elif msg["role"] == "assistant":
                parts = [types.Part.from_text(text=msg["content"])]
                contents.append(types.Content(role="model", parts=parts))

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if system_instruction:
            config.system_instruction = system_instruction

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )

        content = response.text or ""
        logger.info("gemini_call_done", model=self._model, response_len=len(content))
        return content

    async def health_check(self) -> bool:
        try:
            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents="Say OK",
                config=types.GenerateContentConfig(max_output_tokens=5),
            )
            return bool(response.text)
        except Exception:
            return False
