from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    gemini_api_key: str = ""
    groq_api_key: str = ""
    nvidia_api_key: str = ""

    # Models to use (5 diverse models across 3 providers)
    # All tested and confirmed working as of March 2026
    models: list[str] = [
        "groq/openai-gpt-oss-120b",               # Groq — OpenAI open-source 120B
        "groq/llama-4-scout-17b-16e-instruct",    # Groq — Llama 4, different architecture
        "groq/qwen3-32b",                         # Groq — Qwen3, diverse reasoning
        "nvidia/meta/llama-3.3-70b-instruct",     # NVIDIA NIM — fast 70B (405B was timing out)
        "nvidia/google/gemma-2-27b-it",           # NVIDIA NIM — Google architecture
    ]

    # Pipeline tuning
    solver_temperature: float = 0.3
    max_solver_tokens: int = 4096
    parallel_timeout_seconds: float = 90.0   # 90s max per model
    consensus_threshold: float = 0.6  # 3/5 models must agree
    max_debate_rounds: int = 1         # 1 debate round max (was 2)
    max_total_llm_calls: int = 15      # strict budget for resolution phase
    enable_symbolic_verification: bool = True
    min_models_required: int = 2  # minimum successful responses to proceed


settings = Settings()
