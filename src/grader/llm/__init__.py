"""LLM client abstraction: a Protocol plus production and fake implementations."""

from grader.llm.anthropic_client import AnthropicLLMClient
from grader.llm.base import LLMClient, LLMError, LLMRefusalError, LLMResult, LLMUsage
from grader.llm.fake import FakeLLMClient

__all__ = [
    "AnthropicLLMClient",
    "FakeLLMClient",
    "LLMClient",
    "LLMError",
    "LLMRefusalError",
    "LLMResult",
    "LLMUsage",
]
