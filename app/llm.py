"""LLM provider abstraction layer.

The rest of the app only knows about `LLMClient.complete()`. Swapping Claude for
OpenAI (or a fake client in tests) is a config change, not a code change.
Every call returns exact token counts and latency so we can track cost and speed.
"""

import re
import time
from dataclasses import dataclass
from typing import Protocol

from app.config import Settings, resolve_model
from app.prompt import estimate_tokens


class LLMError(RuntimeError):
    """The provider call failed (timeout, auth, rate limit, etc.)."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    provider: str
    model: str


class LLMClient(Protocol):
    def complete(self, system: str, user: str, max_tokens: int) -> LLMResult: ...


class AnthropicClient:
    def __init__(self, model: str, timeout_s: float):
        import anthropic  # imported here so tests never need the SDK or a key

        self.model = model
        self._client = anthropic.Anthropic(timeout=timeout_s, max_retries=2)

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResult:
        start = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # SDK raises several error types; surface one
            raise LLMError(f"Anthropic call failed: {exc}") from exc
        text = "".join(block.text for block in resp.content if block.type == "text")
        return LLMResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            provider="anthropic",
            model=self.model,
        )


class OpenAIClient:
    def __init__(self, model: str, timeout_s: float):
        import openai

        self.model = model
        self._client = openai.OpenAI(timeout=timeout_s, max_retries=2)

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResult:
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as exc:
            raise LLMError(f"OpenAI call failed: {exc}") from exc
        return LLMResult(
            text=resp.choices[0].message.content or "",
            input_tokens=resp.usage.prompt_tokens,
            output_tokens=resp.usage.completion_tokens,
            latency_ms=(time.perf_counter() - start) * 1000,
            provider="openai",
            model=self.model,
        )


class FakeClient:
    """Offline stand-in used by tests, CI and demos without an API key.

    It "answers" by quoting the most relevant source, with a proper citation,
    so the full pipeline (retrieval, prompt, validation, metrics) can be tested.
    """

    model = "fake-echo"

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResult:
        match = re.search(r"\[source (\d+)\] (.+)", user)
        if match:
            text = f"According to the dataset, {match.group(2)} [source {match.group(1)}]."
        else:
            text = "I don't have that in the dataset."
        return LLMResult(
            text=text,
            input_tokens=estimate_tokens(system + user),
            output_tokens=estimate_tokens(text),
            latency_ms=1.0,
            provider="fake",
            model=self.model,
        )


def make_client(settings: Settings) -> LLMClient:
    model = resolve_model(settings)
    provider = settings.llm_provider.lower()
    if provider == "anthropic":
        return AnthropicClient(model, settings.request_timeout_s)
    if provider == "openai":
        return OpenAIClient(model, settings.request_timeout_s)
    if provider == "fake":
        return FakeClient()
    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'")
