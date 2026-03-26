from __future__ import annotations

import structlog

from ..config import settings
from .gemini import GeminiProvider
from .openai_compat import OpenAICompatProvider

logger = structlog.get_logger()

# Provider configurations: model_id -> factory kwargs
PROVIDER_CONFIGS: dict[str, dict] = {
    # ── Groq (OpenAI-compatible, ultra-fast inference) ──
    "groq/llama-3.3-70b-versatile": {
        "type": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "api_key_field": "groq_api_key",
    },
    "groq/openai-gpt-oss-120b": {
        "type": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "openai/gpt-oss-120b",
        "api_key_field": "groq_api_key",
    },
    "groq/llama-4-scout-17b-16e-instruct": {
        "type": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "api_key_field": "groq_api_key",
    },
    "groq/qwen3-32b": {
        "type": "openai_compat",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3-32b",
        "api_key_field": "groq_api_key",
    },
    # ── NVIDIA NIM (OpenAI-compatible) ──
    "nvidia/meta/llama-3.3-70b-instruct": {
        "type": "openai_compat",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.3-70b-instruct",
        "api_key_field": "nvidia_api_key",
    },
    "nvidia/meta/llama-3.1-405b-instruct": {
        "type": "openai_compat",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.1-405b-instruct",
        "api_key_field": "nvidia_api_key",
    },
    "nvidia/google/gemma-2-27b-it": {
        "type": "openai_compat",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "google/gemma-2-27b-it",
        "api_key_field": "nvidia_api_key",
    },
    # ── Gemini (kept for future use when quota resets) ──
    "gemini-2.0-flash": {
        "type": "gemini",
        "model": "gemini-2.0-flash",
    },
}


def build_providers() -> list[GeminiProvider | OpenAICompatProvider]:
    """Build provider instances for all configured models."""
    providers: list[GeminiProvider | OpenAICompatProvider] = []

    for model_id in settings.models:
        config = PROVIDER_CONFIGS.get(model_id)
        if not config:
            logger.warning("unknown_model_id", model_id=model_id)
            continue

        if config["type"] == "gemini":
            if not settings.gemini_api_key:
                logger.warning("missing_api_key", provider="gemini")
                continue
            providers.append(
                GeminiProvider(api_key=settings.gemini_api_key, model=config["model"])
            )

        elif config["type"] == "openai_compat":
            api_key = getattr(settings, config["api_key_field"], "")
            if not api_key:
                logger.warning("missing_api_key", provider=config["api_key_field"])
                continue
            providers.append(
                OpenAICompatProvider(
                    api_key=api_key,
                    base_url=config["base_url"],
                    model=config["model"],
                )
            )

    logger.info("providers_built", count=len(providers), models=[p.model_name for p in providers])
    return providers
