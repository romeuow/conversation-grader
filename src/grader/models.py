"""Domain models shared by the graph, the API and the CLI."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["agent", "customer", "system"]


class Message(BaseModel):
    """One turn of a conversation."""

    model_config = ConfigDict(extra="forbid")

    role: Role
    text: str = Field(min_length=1, max_length=20_000)
    ts: datetime | None = None


class Conversation(BaseModel):
    """A conversation between an agent (human or bot) and a customer."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=128)
    agent_id: str = Field(min_length=1, max_length=128)
    channel: str = Field(default="chat", max_length=32)
    messages: list[Message] = Field(min_length=1)

    def transcript(self) -> str:
        """Render the conversation as `role: text` lines."""
        return "\n".join(f"{m.role}: {m.text}" for m in self.messages)


class CriterionGrade(BaseModel):
    """Structured output produced by the judge for a single criterion.

    This is the schema sent to the LLM via `output_format`; keep it small and stable.
    """

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=5, description="Score on the 0-5 scale defined by the rubric")
    rationale: str = Field(
        min_length=1,
        description="Short justification grounded on the conversation",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Literal excerpts copied verbatim from the conversation",
    )


class CriterionResult(BaseModel):
    """Grade for one criterion after post-validation."""

    criterion_id: str
    criterion_name: str
    weight: float
    score: int = Field(ge=0, le=5)
    rationale: str
    evidence: list[str]
    evidence_verified: bool
    input_tokens: int = 0
    output_tokens: int = 0


class RiskSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class RiskFlag(BaseModel):
    """A compliance or quality risk detected in a conversation."""

    code: str
    severity: RiskSeverity
    description: str
    evidence: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    """Token usage and estimated cost."""

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def add(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            estimated_cost_usd=round(self.estimated_cost_usd + other.estimated_cost_usd, 6),
        )


class ConversationResult(BaseModel):
    """Final evaluation of one conversation."""

    conversation_id: str
    agent_id: str
    channel: str
    rubric_id: str
    rubric_version: str
    overall_score: float = Field(ge=0, le=5, description="Weighted mean on the 0-5 scale")
    criteria: list[CriterionResult]
    risks: list[RiskFlag]
    pii_redactions: dict[str, int]
    usage: TokenUsage
    error: str | None = None


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class JobProgress(BaseModel):
    total: int
    completed: int
    failed: int = 0


class Job(BaseModel):
    """A batch grading job."""

    id: str
    rubric_id: str
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime
    updated_at: datetime
    progress: JobProgress
    results: list[ConversationResult] = Field(default_factory=list)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None


class CriterionAggregate(BaseModel):
    criterion_id: str
    criterion_name: str
    mean_score: float
    n: int


class AgentAggregate(BaseModel):
    agent_id: str
    conversations: int
    mean_overall: float
    by_criterion: dict[str, float]
    risk_count: int


class RiskAggregate(BaseModel):
    code: str
    severity: RiskSeverity
    count: int
    conversation_ids: list[str]


class Report(BaseModel):
    """Aggregated view of a finished (or partially finished) job."""

    job_id: str
    rubric_id: str
    conversations: int
    mean_overall: float
    by_criterion: list[CriterionAggregate]
    by_agent: list[AgentAggregate]
    top_risks: list[RiskAggregate]
    usage: TokenUsage
