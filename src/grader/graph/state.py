"""Typed state for the grading graph."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from grader.models import Conversation, ConversationResult, CriterionResult, RiskFlag, TokenUsage


class GraderState(TypedDict, total=False):
    """State shared across nodes. Lists with reducers accumulate fan-out results."""

    conversation: Conversation
    transcript: str
    redacted_transcript: str
    agent_messages: list[str]
    pii_counts: dict[str, int]
    grades: Annotated[list[CriterionResult], operator.add]
    errors: Annotated[list[str], operator.add]
    risks: list[RiskFlag]
    overall_score: float
    usage: TokenUsage
    result: ConversationResult


class GradeInput(TypedDict):
    """Payload sent to each `grade_criterion` branch via `Send`."""

    conversation_id: str
    criterion_id: str
    redacted_transcript: str
