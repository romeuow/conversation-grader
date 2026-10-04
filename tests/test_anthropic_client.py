"""AnthropicLLMClient is exercised against an in-memory stub of the SDK client. No network."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from grader.llm.anthropic_client import DEFAULT_MODEL, AnthropicLLMClient
from grader.llm.base import LLMError, LLMRefusalError
from grader.models import CriterionGrade


class StubMessages:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict] = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _stub(response):
    return SimpleNamespace(messages=StubMessages(response))


def _response(**overrides):
    base = {
        "stop_reason": "end_turn",
        "parsed_output": CriterionGrade(score=4, rationale="ok", evidence=["Olá"]),
        "usage": SimpleNamespace(input_tokens=321, output_tokens=45),
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_parse_structured_calls_sdk_with_brief_signature():
    stub = _stub(_response())
    client = AnthropicLLMClient(client=stub, model="claude-opus-5-5", max_tokens=16000)
    result = client.parse_structured(system="SYS", user="USER", output_format=CriterionGrade)

    assert client.model == DEFAULT_MODEL == "claude-opus-5-5"
    call = stub.messages.calls[0]
    assert call == {
        "model": "claude-opus-5-5",
        "max_tokens": 16000,
        "system": "SYS",
        "messages": [{"role": "user", "content": "USER"}],
        "output_format": CriterionGrade,
    }
    # The BRIEF forbids these knobs.
    for forbidden in ("temperature", "thinking", "tool_choice", "tools"):
        assert forbidden not in call
    assert result.parsed.score == 4
    assert result.usage.input_tokens == 321 and result.usage.output_tokens == 45
    assert result.model == "claude-opus-5-5"


def test_refusal_is_domain_error():
    client = AnthropicLLMClient(client=_stub(_response(stop_reason="refusal")))
    with pytest.raises(LLMRefusalError):
        client.parse_structured(system="s", user="u", output_format=CriterionGrade)


def test_truncation_is_error():
    client = AnthropicLLMClient(client=_stub(_response(stop_reason="max_tokens")))
    with pytest.raises(LLMError, match="max_tokens"):
        client.parse_structured(system="s", user="u", output_format=CriterionGrade)


def test_missing_parsed_output_is_error():
    client = AnthropicLLMClient(client=_stub(_response(parsed_output=None)))
    with pytest.raises(LLMError, match="no structured output"):
        client.parse_structured(system="s", user="u", output_format=CriterionGrade)


def test_missing_usage_defaults_to_zero():
    client = AnthropicLLMClient(client=_stub(_response(usage=None)))
    result = client.parse_structured(system="s", user="u", output_format=CriterionGrade)
    assert result.usage.input_tokens == 0 and result.usage.output_tokens == 0


def test_default_constructor_uses_anthropic_sdk(monkeypatch: pytest.MonkeyPatch):
    import anthropic

    created: list[dict] = []

    class FakeAnthropic:
        def __init__(self, **kwargs) -> None:
            created.append(kwargs)
            self.messages = StubMessages(_response())

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    client = AnthropicLLMClient()
    assert created == [{}]  # no api_key passed explicitly: the SDK reads the environment
    assert client.model == "claude-opus-5-5"
    result = client.parse_structured(system="s", user="u", output_format=CriterionGrade)
    assert result.parsed.rationale == "ok"
