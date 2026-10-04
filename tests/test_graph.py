"""Integration tests for the full LangGraph pipeline (fake judge)."""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from grader.graph.build import ConversationGrader, build_graph
from grader.graph.nodes import Pricing
from grader.graph.state import GraderState
from grader.llm.base import LLMError, LLMRefusalError, LLMResult, LLMUsage
from grader.models import Conversation, ConversationResult, CriterionGrade, Message


class FabricatingLLM:
    """Returns evidence that does not exist in the transcript."""

    model = "fabricator"

    def parse_structured(self, *, system, user, output_format):
        return LLMResult(
            parsed=CriterionGrade(score=4, rationale="ok", evidence=["frase inventada"]),
            usage=LLMUsage(input_tokens=100, output_tokens=10),
            model=self.model,
        )


class FailingLLM:
    model = "failing"

    def __init__(self, fail_on: set[str]) -> None:
        self.fail_on = fail_on

    def parse_structured(self, *, system, user, output_format):
        if any(f'<criterion id="{cid}"' in user for cid in self.fail_on):
            raise LLMRefusalError("refused")
        return LLMResult(
            parsed=CriterionGrade(score=2, rationale="meh", evidence=["Olá"]),
            usage=LLMUsage(input_tokens=1, output_tokens=1),
            model=self.model,
        )


def test_full_pipeline_good_conversation(grader: ConversationGrader, by_id, rubric, fake_llm):
    result = grader.grade(by_id["conv-001"])
    assert isinstance(result, ConversationResult)
    assert result.conversation_id == "conv-001"
    assert result.rubric_id == "sales_v1" and result.rubric_version == "1.0.0"
    assert [c.criterion_id for c in result.criteria] == [c.id for c in rubric.criteria]
    assert all(c.evidence_verified for c in result.criteria)
    assert result.overall_score == 5.0
    assert result.risks == [] and result.pii_redactions == {} and result.error is None
    assert fake_llm.calls == len(rubric.criteria)
    assert result.usage.input_tokens > 0 and result.usage.estimated_cost_usd > 0


def test_pipeline_redacts_pii_before_llm(rubric, by_id, settings):
    seen: list[str] = []

    class SpyLLM:
        model = "spy"

        def parse_structured(self, *, system, user, output_format):
            seen.append(user)
            return LLMResult(
                parsed=CriterionGrade(score=3, rationale="r", evidence=["[CPF]"]),
                usage=LLMUsage(),
                model=self.model,
            )

    result = ConversationGrader(SpyLLM(), rubric, settings=settings).grade(by_id["conv-004"])
    assert seen and all("000.000.000-00" not in prompt for prompt in seen)
    assert all("[CPF]" in prompt and "[PHONE]" in prompt and "[EMAIL]" in prompt for prompt in seen)
    assert result.pii_redactions == {"CPF": 1, "PHONE": 1, "EMAIL": 1}
    assert [r.code for r in result.risks] == ["pii_shared"]


def test_bad_conversation_raises_risks(grader: ConversationGrader, by_id):
    result = grader.grade(by_id["conv-003"])
    codes = {r.code for r in result.risks}
    assert {"undue_promise", "artificial_urgency"} <= codes
    compliance = next(c for c in result.criteria if c.criterion_id == "compliance")
    assert compliance.score == 0
    assert result.overall_score < 3


def test_fabricated_evidence_is_flagged(rubric, by_id, settings):
    result = ConversationGrader(FabricatingLLM(), rubric, settings=settings).grade(
        by_id["conv-001"]
    )
    assert all(c.evidence_verified is False for c in result.criteria)
    assert all(c.score == 4 for c in result.criteria)
    # 5 criteria x (100 in + 10 out) at default pricing 4/20 per MTok
    assert result.usage.input_tokens == 500 and result.usage.output_tokens == 50
    assert result.usage.estimated_cost_usd == pytest.approx((500 * 4 + 50 * 20) / 1e6)


def test_llm_error_is_recorded_not_raised(rubric, by_id, settings):
    llm = FailingLLM(fail_on={"compliance"})
    result = ConversationGrader(llm, rubric, settings=settings).grade(by_id["conv-001"])
    assert result.error == "compliance: refused"
    assert [c.criterion_id for c in result.criteria] == [
        "opening",
        "discovery",
        "objection_handling",
        "next_steps",
    ]
    assert result.overall_score == 2.0


def test_generic_llm_error_is_llm_error():
    assert issubclass(LLMRefusalError, LLMError)


def test_checkpointer_records_state(rubric, fake_llm, settings, by_id):
    saver = InMemorySaver()
    grader = ConversationGrader(fake_llm, rubric, settings=settings, checkpointer=saver)
    grader.grade(by_id["conv-002"])
    snapshot = grader.graph.get_state({"configurable": {"thread_id": "conv-002"}})
    assert snapshot.next == ()
    assert snapshot.values["result"].conversation_id == "conv-002"
    assert set(snapshot.values) >= {"transcript", "redacted_transcript", "grades", "risks"}


async def test_async_invocation(grader: ConversationGrader, by_id):
    result = await grader.agrade(by_id["conv-005"])
    assert result.agent_id == "bot-funil-v2"
    assert result.overall_score > 3


def test_build_graph_with_explicit_pricing(rubric, fake_llm, by_id):
    graph = build_graph(fake_llm, rubric, pricing=Pricing(0.0, 0.0))
    state: GraderState = graph.invoke({"conversation": by_id["conv-001"]})
    assert state["result"].usage.estimated_cost_usd == 0.0
    assert state["grades"] and state["overall_score"] == 5.0


def test_normalize_collapses_whitespace(grader: ConversationGrader):
    conversation = Conversation(
        id="ws",
        agent_id="a",
        messages=[Message(role="agent", text="  Olá,    bom   dia!  Meu nome é  Ana. Tudo bem?  ")],
    )
    state = grader.graph.invoke({"conversation": conversation})
    assert state["transcript"] == "agent: Olá, bom dia! Meu nome é Ana. Tudo bem?"
    assert state["agent_messages"] == ["Olá, bom dia! Meu nome é Ana. Tudo bem?"]
