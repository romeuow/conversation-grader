from grader.models import CriterionResult, RiskSeverity
from grader.risks import detect_risks


def _grade(cid: str, score: int) -> CriterionResult:
    return CriterionResult(
        criterion_id=cid,
        criterion_name=cid,
        weight=1,
        score=score,
        rationale="r",
        evidence=["e"],
        evidence_verified=True,
    )


def test_no_risks_for_clean_conversation():
    assert detect_risks(["Olá, posso ajudar?"], {}, [_grade("compliance", 5)]) == []


def test_lexical_rules():
    risks = detect_risks(
        ["Eu garanto 30% de economia", "A oferta é só hoje", "Me passa sua senha"], {}, []
    )
    codes = {r.code: r for r in risks}
    assert set(codes) == {"undue_promise", "artificial_urgency", "sensitive_data_request"}
    assert codes["undue_promise"].severity == RiskSeverity.HIGH
    assert codes["artificial_urgency"].severity == RiskSeverity.MEDIUM
    assert codes["undue_promise"].evidence == ["Eu garanto 30% de economia"]


def test_pii_severity_depends_on_kind():
    assert detect_risks([], {"EMAIL": 1}, [])[0].severity == RiskSeverity.LOW
    assert detect_risks([], {"CPF": 1, "EMAIL": 1}, [])[0].severity == RiskSeverity.MEDIUM
    risk = detect_risks([], {"CARD": 1}, [])[0]
    assert risk.severity == RiskSeverity.HIGH
    assert risk.code == "pii_shared"
    assert "CARD=1" in risk.description


def test_low_compliance_score_only_when_not_already_flagged():
    low = [_grade("compliance", 1)]
    assert [r.code for r in detect_risks(["tudo certo"], {}, low)] == ["low_compliance_score"]
    flagged = detect_risks(["Eu garanto tudo"], {}, low)
    assert [r.code for r in flagged] == ["undue_promise"]
    assert detect_risks(["ok"], {}, [_grade("compliance", 2)]) == []
