"""Settings, read once from environment variables (see .env.example)."""

import os
from dataclasses import dataclass, field


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # Which LLM provider to call: "anthropic", "openai" or "fake" (offline, for tests/demos).
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "fake"))
    # Model name for the chosen provider. Left configurable so it never goes stale in code.
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", ""))

    # Path to the CSV dataset. Every row becomes one searchable "document".
    data_path: str = field(
        default_factory=lambda: os.getenv("DATA_PATH", "data/forest_carbon_long.csv")
    )

    # Token management: how much retrieved context we send, and how long an answer may be.
    max_context_tokens: int = field(default_factory=lambda: _int("MAX_CONTEXT_TOKENS", 1500))
    max_output_tokens: int = field(default_factory=lambda: _int("MAX_OUTPUT_TOKENS", 400))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 25))

    # Latency control: give up on a slow provider call after this many seconds.
    request_timeout_s: float = field(default_factory=lambda: _float("REQUEST_TIMEOUT_S", 30.0))

    # Cost tracking: USD per million tokens. Set these from your provider's pricing page.
    cost_per_mtok_input: float = field(default_factory=lambda: _float("COST_PER_MTOK_INPUT", 0.0))
    cost_per_mtok_output: float = field(default_factory=lambda: _float("COST_PER_MTOK_OUTPUT", 0.0))


DEFAULT_MODELS = {
    "anthropic": "claude-haiku-5-5",
    "openai": "gpt-4o-mini",
    "fake": "fake-echo",
}


def resolve_model(settings: Settings) -> str:
    return settings.llm_model or DEFAULT_MODELS.get(settings.llm_provider, "")
