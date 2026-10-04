"""Deterministic risk detection applied on top of the judge's grades.

Risks are rule-based on purpose: they must be auditable, cheap, and independent from the
LLM so that an alert is never missed because of a model regression.
"""

from __future__ import annotations

import re

from grader.models import CriterionResult, RiskFlag, RiskSeverity

_RULES: list[tuple[str, RiskSeverity, str, re.Pattern[str]]] = [
    (
        "undue_promise",
        RiskSeverity.HIGH,
        "Promessa de resultado, garantia ou isencao sem base contratual",
        re.compile(
            r"\b(garant\w*|100 ?%|sem (nenhum )?risco|zero risco|isen[cç]\w*|nunca falha)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "artificial_urgency",
        RiskSeverity.MEDIUM,
        "Pressao com urgencia artificial",
        re.compile(r"\b(s[oó] hoje|[uú]ltima chance|agora ou nunca)\b", re.IGNORECASE),
    ),
    (
        "sensitive_data_request",
        RiskSeverity.HIGH,
        "Solicitacao de senha ou dados de cartao",
        re.compile(r"\b(senha|cvv|cart[aã]o de cr[eé]dito)\b", re.IGNORECASE),
    ),
]


def detect_risks(
    agent_messages: list[str],
    pii_counts: dict[str, int],
    grades: list[CriterionResult],
) -> list[RiskFlag]:
    """Combine lexical rules, PII counters and low compliance grades into risk flags."""
    risks: list[RiskFlag] = []
    for code, severity, description, pattern in _RULES:
        hits = [m for m in agent_messages if pattern.search(m)]
        if hits:
            risks.append(
                RiskFlag(
                    code=code,
                    severity=severity,
                    description=description,
                    evidence=[h[:160] for h in hits[:3]],
                )
            )
    if pii_counts:
        kinds = ", ".join(f"{k}={v}" for k, v in sorted(pii_counts.items()))
        severity = (
            RiskSeverity.HIGH
            if "CARD" in pii_counts
            else RiskSeverity.MEDIUM
            if "CPF" in pii_counts
            else RiskSeverity.LOW
        )
        risks.append(
            RiskFlag(
                code="pii_shared",
                severity=severity,
                description=f"Dados pessoais trafegaram na conversa ({kinds})",
            )
        )
    for grade in grades:
        is_low_compliance = grade.criterion_id == "compliance" and grade.score <= 1
        if is_low_compliance and not any(r.code == "undue_promise" for r in risks):
            risks.append(
                RiskFlag(
                    code="low_compliance_score",
                    severity=RiskSeverity.HIGH,
                    description="Juiz atribuiu nota baixa em conformidade",
                    evidence=grade.evidence[:3],
                )
            )
    return risks
