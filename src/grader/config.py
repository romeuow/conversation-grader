"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration. Every field can be overridden via env vars."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "conversation-grader"
    log_level: str = "INFO"

    # LLM
    llm_model: str = Field(default="claude-opus-5-5", alias="LLM_MODEL")
    llm_max_tokens: int = 16000
    use_fake_llm: bool = Field(default=True, alias="USE_FAKE_LLM")
    # Price per million tokens used for cost estimation (USD).
    llm_input_price_per_mtok: float = 4.0
    llm_output_price_per_mtok: float = 20.0

    # Grading
    rubrics_dir: Path = PROJECT_ROOT / "rubrics"
    max_concurrency: int = 4

    # Security: when set, every API request must send `X-API-Key: <value>`.
    api_key: str | None = Field(default=None, alias="API_KEY")

    # Limits
    max_conversations_per_job: int = 200
    max_messages_per_conversation: int = 500


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
