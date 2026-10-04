"""Production LLM client backed by the official `anthropic` SDK."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from grader.llm.base import LLMError, LLMRefusalError, LLMResult, LLMUsage

if TYPE_CHECKING:
    import anthropic

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicLLMClient:
    """Structured-output client using `client.messages.parse(..., output_format=Model)`.

    The `anthropic.Anthropic()` constructor reads `ANTHROPIC_API_KEY` from the environment;
    the key is never handled by this code. A pre-built client can be injected for tests.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 16000,
        client: anthropic.Anthropic | Any | None = None,
    ) -> None:
        if client is None:
            import anthropic as _anthropic

            client = _anthropic.Anthropic()
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def parse_structured[T: BaseModel](
        self, *, system: str, user: str, output_format: type[T]
    ) -> LLMResult[T]:
        response = self._client.messages.parse(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
        )
        if response.stop_reason == "refusal":
            raise LLMRefusalError("model refused to grade this conversation")
        if response.stop_reason == "max_tokens":
            raise LLMError("response truncated by max_tokens")
        parsed = response.parsed_output
        if parsed is None:
            raise LLMError("model returned no structured output")
        usage = getattr(response, "usage", None)
        return LLMResult(
            parsed=parsed,
            usage=LLMUsage(
                input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            ),
            model=self._model,
        )
