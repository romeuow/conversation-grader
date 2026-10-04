"""Node implementations. Each node is a pure-ish function over the typed state."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from langgraph.types import Send

from grader.aggregate import estimate_cost, weighted_score
from grader.evidence import verify_evidence
from grader.graph.state import GradeInput, GraderState
from grader.llm.base import LLMClient, LLMError
from grader.log import get_logger
from grader.models import ConversationResult, CriterionGrade, CriterionResult, TokenUsage
from grader.pii import redact_pii
from grader.prompts import SYSTEM_PROMPT, build_user_prompt
from grader.risks import detect_risks
from grader.rubric import Rubric

log = get_logger(__name__)

_WS = re.compile(r"[ \t]+")


@dataclass(frozen=True)
class Pricing:
    input_per_mtok: float
    output_per_mtok: float


class GraphNodes:
    """Holds the dependencies (LLM, rubric, pricing) the nodes need."""

    def __init__(self, llm: LLMClient, rubric: Rubric, pricing: Pricing) -> None:
        self.llm = llm
        self.rubric = rubric
        self.pricing = pricing

    # 1. normalize ------------------------------------------------------------------
    def normalize(self, state: GraderState) -> GraderState:
        conversation = state["conversation"]
        cleaned: list[str] = []
        for message in conversation.messages:
            text = _WS.sub(" ", message.text).strip()
            cleaned.append(f"{message.role}: {text}")
        return {"transcript": "\n".join(cleaned)}

    # 2. redact_pii -----------------------------------------------------------------
    def redact_pii(self, state: GraderState) -> GraderState:
        redaction = redact_pii(state["transcript"])
        agent_messages = [
            line.split(": ", 1)[1]
            for line in redaction.text.splitlines()
            if line.startswith("agent: ")
        ]
        if redaction.total:
            log.info(
                "pii_redacted",
                extra={
                    "conversation_id": state["conversation"].id,
                    "counts": redaction.counts,
                },
            )
        return {
            "redacted_transcript": redaction.text,
            "pii_counts": redaction.counts,
            "agent_messages": agent_messages,
        }

    # 3. fan-out --------------------------------------------------------------------
    def fan_out_criteria(self, state: GraderState) -> list[Send]:
        """Create one `grade_criterion` branch per rubric criterion."""
        return [
            Send(
                "grade_criterion",
                GradeInput(
                    conversation_id=state["conversation"].id,
                    criterion_id=criterion.id,
                    redacted_transcript=state["redacted_transcript"],
                ),
            )
            for criterion in self.rubric.criteria
        ]

    def grade_criterion(self, payload: GradeInput) -> GraderState:
        criterion = self.rubric.get(payload["criterion_id"])
        transcript = payload["redacted_transcript"]
        user_prompt = build_user_prompt(self.rubric, criterion, transcript)
        started = time.perf_counter()
        try:
            result = self.llm.parse_structured(
                system=SYSTEM_PROMPT, user=user_prompt, output_format=CriterionGrade
            )
        except LLMError as exc:
            log.warning(
                "grade_failed",
                extra={
                    "conversation_id": payload["conversation_id"],
                    "criterion_id": criterion.id,
                    "error": str(exc),
                },
            )
            return {"errors": [f"{criterion.id}: {exc}"]}
        grade = result.parsed
        verified = verify_evidence(grade.evidence, transcript)
        log.info(
            "criterion_graded",
            extra={
                "conversation_id": payload["conversation_id"],
                "criterion_id": criterion.id,
                "score": grade.score,
                "evidence_verified": verified,
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                "model": result.model,
            },
        )
        return {
            "grades": [
                CriterionResult(
                    criterion_id=criterion.id,
                    criterion_name=criterion.name,
                    weight=criterion.weight,
                    score=grade.score,
                    rationale=grade.rationale,
                    evidence=grade.evidence,
                    evidence_verified=verified,
                    input_tokens=result.usage.input_tokens,
                    output_tokens=result.usage.output_tokens,
                )
            ]
        }

    # 4. detect_risks ---------------------------------------------------------------
    def detect_risks(self, state: GraderState) -> GraderState:
        risks = detect_risks(
            state.get("agent_messages", []), state.get("pii_counts", {}), state.get("grades", [])
        )
        return {"risks": risks}

    # 5. aggregate ------------------------------------------------------------------
    def aggregate(self, state: GraderState) -> GraderState:
        grades = sorted(state.get("grades", []), key=self._criterion_order)
        input_tokens = sum(g.input_tokens for g in grades)
        output_tokens = sum(g.output_tokens for g in grades)
        usage = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimate_cost(
                input_tokens,
                output_tokens,
                input_price=self.pricing.input_per_mtok,
                output_price=self.pricing.output_per_mtok,
            ),
        )
        return {"grades": [], "overall_score": weighted_score(grades), "usage": usage}

    # 6. finalize -------------------------------------------------------------------
    def finalize(self, state: GraderState) -> GraderState:
        conversation = state["conversation"]
        grades = sorted(state.get("grades", []), key=self._criterion_order)
        errors = state.get("errors", [])
        result = ConversationResult(
            conversation_id=conversation.id,
            agent_id=conversation.agent_id,
            channel=conversation.channel,
            rubric_id=self.rubric.id,
            rubric_version=self.rubric.version,
            overall_score=state.get("overall_score", 0.0),
            criteria=grades,
            risks=state.get("risks", []),
            pii_redactions=state.get("pii_counts", {}),
            usage=state.get("usage", TokenUsage()),
            error="; ".join(errors) if errors else None,
        )
        log.info(
            "conversation_graded",
            extra={
                "conversation_id": conversation.id,
                "agent_id": conversation.agent_id,
                "overall_score": result.overall_score,
                "risks": [r.code for r in result.risks],
                "errors": len(errors),
            },
        )
        return {"result": result}

    def _criterion_order(self, grade: CriterionResult) -> int:
        for index, criterion in enumerate(self.rubric.criteria):
            if criterion.id == grade.criterion_id:
                return index
        return len(self.rubric.criteria)
