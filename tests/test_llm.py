"""Check each provider adapter parses responses correctly, without calling the real APIs."""

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.llm import AnthropicClient, FakeClient, LLMError, OpenAIClient, make_client


class StubCreate:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def test_anthropic_adapter_reads_text_and_token_usage(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient("claude-haiku-5-5", timeout_s=5)
    stub = StubCreate(SimpleNamespace(
        content=[SimpleNamespace(type="text", text="Answer [source 1].")],
        usage=SimpleNamespace(input_tokens=120, output_tokens=8),
    ))
    client._client = SimpleNamespace(messages=stub)

    result = client.complete("system prompt", "question", max_tokens=50)

    assert result.text == "Answer [source 1]."
    assert (result.input_tokens, result.output_tokens) == (120, 8)
    assert stub.kwargs["max_tokens"] == 50 and stub.kwargs["system"] == "system prompt"


def test_openai_adapter_reads_text_and_token_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    client = OpenAIClient("gpt-4o-mini", timeout_s=5)
    stub = StubCreate(SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="Answer [source 2]."))],
        usage=SimpleNamespace(prompt_tokens=90, completion_tokens=6),
    ))
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=stub))

    result = client.complete("system prompt", "question", max_tokens=50)

    assert result.text == "Answer [source 2]."
    assert (result.input_tokens, result.output_tokens) == (90, 6)


def test_provider_errors_become_llm_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient("claude-haiku-5-5", timeout_s=5)
    client._client = SimpleNamespace(messages=StubCreate(error=TimeoutError("slow")))
    with pytest.raises(LLMError):
        client.complete("s", "q", max_tokens=10)


def test_factory_picks_provider_from_config():
    assert isinstance(make_client(Settings(llm_provider="fake")), FakeClient)
    with pytest.raises(ValueError):
        make_client(Settings(llm_provider="nope"))
