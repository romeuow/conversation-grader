import pytest

from grader.aggregate import build_report, estimate_cost, weighted_score
from grader.models import (
    ConversationResult,
    CriterionResult,
    RiskFlag,
    RiskSeverity,
    TokenUsage,
)


def _grade(cid: str, score: int, weight: float = 1.0) -> CriterionResult:
    return CriterionResult(
        criterion_id=cid,
        criterion_name=cid.title(),
        weight=weight,
        score=score,
        rationale="r",
        evidence=["e"],
        evidence_verified=True,
        input_tokens=10,
        output_tokens=5,
    )


def _result(cid: str, agent: str, grades: list[CriterionResult], risks=(), error=None):
    return ConversationResult(
        conversation_id=cid,
        agent_id=agent,
        channel="chat",
        rubric_id="sales_v1",
        rubric_version="1.0.0",
        overall_score=weighted_score(grades),
        criteria=grades,
        risks=list(risks),
        pii_redactions={},
        usage=TokenUsage(input_tokens=10, output_tokens=5, estimated_cost_usd=0.001),
        error=error,
    )


def test_weighted_score():
    assert weighted_score([]) == 0.0
    assert weighted_score([_grade("a", 5), _grade("b", 1)]) == 3.0
    assert weighted_score([_grade("a", 5, weight=3), _grade("b", 1, weight=1)]) == 4.0


def test_estimate_cost_uses_per_million_prices():
    assert estimate_cost(1_000_000, 0, input_price=4.0, output_price=20.0) == 4.0
    assert estimate_cost(500_000, 100_000, input_price=4.0, output_price=20.0) == pytest.approx(4.0)


def test_token_usage_add():
    total = TokenUsage(input_tokens=1, output_tokens=2, estimated_cost_usd=0.5).add(
        TokenUsage(input_tokens=3, output_tokens=4, estimated_cost_usd=0.25)
    )
    assert (total.input_tokens, total.output_tokens, total.estimated_cost_usd) == (4, 6, 0.75)


def test_build_report_aggregates_and_ranks():
    high = RiskFlag(code="undue_promise", severity=RiskSeverity.HIGH, description="d")
    low = RiskFlag(code="pii_shared", severity=RiskSeverity.LOW, description="d")
    results = [
        _result("c1", "alice", [_grade("opening", 5), _grade("compliance", 5)]),
        _result("c2", "bob", [_grade("opening", 1), _grade("compliance", 1)], risks=[high, low]),
        _result("c3", "bob", [_grade("opening", 3), _grade("compliance", 3)], risks=[low]),
        _result("c4", "carol", [], error="LLMError: boom"),
    ]
    usage = TokenUsage(input_tokens=30, output_tokens=15, estimated_cost_usd=0.003)
    report = build_report("job-1", "sales_v1", results, usage)

    assert report.conversations == 3  # failed one excluded
    assert report.mean_overall == pytest.approx((5 + 1 + 3) / 3, abs=1e-3)
    assert {c.criterion_id: c.mean_score for c in report.by_criterion} == {
        "opening": 3.0,
        "compliance": 3.0,
    }
    assert [a.agent_id for a in report.by_agent] == ["alice", "bob"]
    bob = report.by_agent[1]
    assert bob.conversations == 2 and bob.mean_overall == 2.0 and bob.risk_count == 3
    assert bob.by_criterion == {"opening": 2.0, "compliance": 2.0}
    # HIGH severity first, then by count
    assert [(r.code, r.count) for r in report.top_risks] == [
        ("undue_promise", 1),
        ("pii_shared", 2),
    ]
    assert report.top_risks[1].conversation_ids == ["c2", "c3"]
    assert report.usage == usage


def test_build_report_empty():
    report = build_report("job", "sales_v1", [], TokenUsage())
    assert report.conversations == 0 and report.mean_overall == 0.0
    assert report.by_agent == [] and report.top_risks == []
