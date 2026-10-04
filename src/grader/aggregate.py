"""Aggregation helpers: per-conversation score and per-job report."""

from __future__ import annotations

from collections import defaultdict
from statistics import fmean

from grader.models import (
    AgentAggregate,
    ConversationResult,
    CriterionAggregate,
    CriterionResult,
    Report,
    RiskAggregate,
    RiskSeverity,
    TokenUsage,
)

_SEVERITY_ORDER = {RiskSeverity.HIGH: 0, RiskSeverity.MEDIUM: 1, RiskSeverity.LOW: 2}


def weighted_score(grades: list[CriterionResult]) -> float:
    """Weighted mean of criterion scores on the 0-5 scale (0.0 when no grades)."""
    total_weight = sum(g.weight for g in grades)
    if total_weight == 0:
        return 0.0
    return round(sum(g.score * g.weight for g in grades) / total_weight, 3)


def estimate_cost(
    input_tokens: int, output_tokens: int, *, input_price: float, output_price: float
) -> float:
    """USD cost given per-million-token prices."""
    return round((input_tokens * input_price + output_tokens * output_price) / 1_000_000, 6)


def build_report(
    job_id: str,
    rubric_id: str,
    results: list[ConversationResult],
    usage: TokenUsage,
    *,
    top_n: int = 5,
) -> Report:
    """Aggregate finished conversation results into a job-level report."""
    ok = [r for r in results if r.error is None]

    criterion_scores: dict[str, list[int]] = defaultdict(list)
    criterion_names: dict[str, str] = {}
    for result in ok:
        for grade in result.criteria:
            criterion_scores[grade.criterion_id].append(grade.score)
            criterion_names[grade.criterion_id] = grade.criterion_name
    by_criterion = [
        CriterionAggregate(
            criterion_id=cid,
            criterion_name=criterion_names[cid],
            mean_score=round(fmean(scores), 3),
            n=len(scores),
        )
        for cid, scores in criterion_scores.items()
    ]

    per_agent: dict[str, list[ConversationResult]] = defaultdict(list)
    for result in ok:
        per_agent[result.agent_id].append(result)
    by_agent: list[AgentAggregate] = []
    for agent_id, agent_results in per_agent.items():
        agent_criteria: dict[str, list[int]] = defaultdict(list)
        for result in agent_results:
            for grade in result.criteria:
                agent_criteria[grade.criterion_id].append(grade.score)
        by_agent.append(
            AgentAggregate(
                agent_id=agent_id,
                conversations=len(agent_results),
                mean_overall=round(fmean(r.overall_score for r in agent_results), 3),
                by_criterion={cid: round(fmean(s), 3) for cid, s in agent_criteria.items()},
                risk_count=sum(len(r.risks) for r in agent_results),
            )
        )
    by_agent.sort(key=lambda a: a.mean_overall, reverse=True)

    risk_index: dict[str, RiskAggregate] = {}
    for result in ok:
        for risk in result.risks:
            agg = risk_index.get(risk.code)
            if agg is None:
                agg = RiskAggregate(
                    code=risk.code, severity=risk.severity, count=0, conversation_ids=[]
                )
                risk_index[risk.code] = agg
            agg.count += 1
            agg.conversation_ids.append(result.conversation_id)
    top_risks = sorted(risk_index.values(), key=lambda r: (_SEVERITY_ORDER[r.severity], -r.count))[
        :top_n
    ]

    return Report(
        job_id=job_id,
        rubric_id=rubric_id,
        conversations=len(ok),
        mean_overall=round(fmean(r.overall_score for r in ok), 3) if ok else 0.0,
        by_criterion=by_criterion,
        by_agent=by_agent,
        top_risks=top_risks,
        usage=usage,
    )
