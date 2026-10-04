"""LLM client protocol and shared result types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base class for LLM failures that are not transport errors."""


class LLMRefusalError(LLMError):
    """The model declined to answer (`stop_reason == "refusal"`)."""


@dataclass(frozen=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class LLMResult[T: BaseModel]:
    parsed: T
    usage: LLMUsage
    model: str


class LLMClient(Protocol):
    """Anything that can turn a prompt into a validated pydantic object."""

    @property
    def model(self) -> str: ...

    def parse_structured(self, *, system: str, user: str, output_format: type[T]) -> LLMResult[T]:
        """Run one structured-output request and return the parsed model."""
        ...
