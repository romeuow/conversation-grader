"""Prompt regression suite: expected scores per criterion with a tolerance."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from grader.graph.build import ConversationGrader
from grader.models import Conversation


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    description: str = ""
    conversation: Conversation
    expected: dict[str, int] = Field(min_length=1)
    tolerance: int | None = Field(default=None, ge=0, le=5)


class EvalSuite(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rubric_id: str
    default_tolerance: int = Field(default=1, ge=0, le=5)
    cases: list[EvalCase] = Field(min_length=1)


class CriterionCheck(BaseModel):
    criterion_id: str
    expected: int
    actual: int | None
    tolerance: int
    passed: bool


class CaseResult(BaseModel):
    case_id: str
    checks: list[CriterionCheck]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)


class EvalReport(BaseModel):
    rubric_id: str
    model: str
    cases: list[CaseResult]

    @property
    def total_checks(self) -> int:
        return sum(len(c.checks) for c in self.cases)

    @property
    def failed_checks(self) -> list[tuple[str, CriterionCheck]]:
        return [(c.case_id, chk) for c in self.cases for chk in c.checks if not chk.passed]

    @property
    def passed(self) -> bool:
        return not self.failed_checks


def load_eval_suite(path: Path) -> EvalSuite:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return EvalSuite.model_validate(raw)


def run_eval_suite(suite: EvalSuite, grader: ConversationGrader) -> EvalReport:
    """Grade every case and compare each expected criterion score within tolerance."""
    results: list[CaseResult] = []
    for case in suite.cases:
        tolerance = suite.default_tolerance if case.tolerance is None else case.tolerance
        outcome = grader.grade(case.conversation)
        actual_by_id = {g.criterion_id: g.score for g in outcome.criteria}
        checks = []
        for criterion_id, expected in case.expected.items():
            actual = actual_by_id.get(criterion_id)
            passed = actual is not None and abs(actual - expected) <= tolerance
            checks.append(
                CriterionCheck(
                    criterion_id=criterion_id,
                    expected=expected,
                    actual=actual,
                    tolerance=tolerance,
                    passed=passed,
                )
            )
        results.append(CaseResult(case_id=case.id, checks=checks))
    return EvalReport(rubric_id=suite.rubric_id, model=grader.llm.model, cases=results)
