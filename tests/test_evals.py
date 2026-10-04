from pathlib import Path

import pytest

from grader.config import PROJECT_ROOT
from grader.evals import EvalSuite, load_eval_suite, run_eval_suite

SUITE = PROJECT_ROOT / "evals" / "sales_v1.yaml"


def test_load_suite():
    suite = load_eval_suite(SUITE)
    assert suite.rubric_id == "sales_v1"
    assert suite.default_tolerance == 1
    assert len(suite.cases) >= 5


def test_suite_passes_with_fake_judge(grader):
    report = run_eval_suite(load_eval_suite(SUITE), grader)
    assert report.model == "fake-judge-v1"
    assert report.failed_checks == []
    assert report.passed and report.total_checks >= 10


def test_regression_is_detected(grader, tmp_path: Path):
    suite = EvalSuite.model_validate(
        {
            "rubric_id": "sales_v1",
            "default_tolerance": 0,
            "cases": [
                {
                    "id": "impossible",
                    "conversation": {
                        "id": "x",
                        "agent_id": "a",
                        "messages": [{"role": "agent", "text": "Quer contratar?"}],
                    },
                    "expected": {"opening": 5, "unknown_criterion": 3},
                }
            ],
        }
    )
    report = run_eval_suite(suite, grader)
    assert not report.passed
    failed = {chk.criterion_id: chk for _, chk in report.failed_checks}
    assert failed["opening"].actual == 0 and failed["opening"].expected == 5
    assert failed["unknown_criterion"].actual is None


def test_suite_validation_rejects_unknown_fields():
    with pytest.raises(ValueError):
        EvalSuite.model_validate({"rubric_id": "r", "cases": [], "bogus": 1})
