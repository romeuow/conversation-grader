"""Shared fixtures. Everything runs offline against the FakeLLMClient."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from grader.config import PROJECT_ROOT, Settings
from grader.graph.build import ConversationGrader
from grader.llm.fake import FakeLLMClient
from grader.models import Conversation
from grader.rubric import Rubric, RubricRegistry, load_rubric

RUBRICS_DIR = PROJECT_ROOT / "rubrics"
EXAMPLES = PROJECT_ROOT / "examples" / "conversations.jsonl"


@pytest.fixture(scope="session")
def rubric() -> Rubric:
    return load_rubric(RUBRICS_DIR / "sales_v1.yaml")


@pytest.fixture(scope="session")
def registry() -> RubricRegistry:
    return RubricRegistry(RUBRICS_DIR)


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        USE_FAKE_LLM=True,
        rubrics_dir=RUBRICS_DIR,
        max_concurrency=2,
        log_level="WARNING",
    )


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def grader(fake_llm: FakeLLMClient, rubric: Rubric, settings: Settings) -> ConversationGrader:
    return ConversationGrader(fake_llm, rubric, settings=settings)


@pytest.fixture(scope="session")
def conversations() -> list[Conversation]:
    lines = EXAMPLES.read_text(encoding="utf-8").splitlines()
    return [Conversation.model_validate(json.loads(line)) for line in lines if line.strip()]


@pytest.fixture(scope="session")
def by_id(conversations: list[Conversation]) -> dict[str, Conversation]:
    return {c.id: c for c in conversations}


@pytest.fixture
def tmp_rubric_file(tmp_path: Path) -> Path:
    path = tmp_path / "mini.yaml"
    path.write_text(
        """
id: mini_v1
version: "0.1"
name: Mini
criteria:
  - id: opening
    name: Abertura
    description: Saudacao
    weight: 1
    anchors: {0: nada, 5: tudo}
""",
        encoding="utf-8",
    )
    return path
