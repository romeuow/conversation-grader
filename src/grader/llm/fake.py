"""Deterministic fake judge used in tests and demo mode.

It parses the same prompt sent to the real model, extracts the criterion id and the
transcript, and applies simple lexical heuristics to produce plausible grades with
literal evidence. No network, no randomness.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from grader.llm.base import LLMError, LLMResult, LLMUsage
from grader.models import CriterionGrade

_CRITERION_RE = re.compile(r'<criterion id="([a-z0-9_]+)"')
_TRANSCRIPT_RE = re.compile(r"<transcript>\n(.*?)\n</transcript>", re.DOTALL)
_LINE_RE = re.compile(r"^(agent|customer|system): (.*)$")

GREETINGS = ("olá", "ola", "bom dia", "boa tarde", "boa noite", "tudo bem", "prazer")
INTRO = (
    "meu nome",
    "me chamo",
    "sou da",
    "sou do",
    "sou o",
    "sou a",
    "falo da",
    "falo do",
    "aqui é",
    "falando",
)
OBJECTIONS = ("caro", "não sei", "nao sei", "pensar", "não tenho certeza", "desconfi", "medo")
ACKNOWLEDGE = ("entendo", "compreendo", "faz sentido", "boa pergunta", "é natural", "e natural")
ALTERNATIVES = ("alternativa", "outra opção", "podemos", "sem compromisso", "dados", "simula")
NEXT_STEPS = (
    "agend",
    "proposta",
    "próximo passo",
    "proximo passo",
    "amanhã",
    "amanha",
    "retorno",
    "envio",
    "enviar",
)
TIME_HINT = ("às", "as ", "h ", "horas", "segunda", "terça", "quarta", "quinta", "sexta", "dia ")
BAD_PROMISES = (
    "garanto",
    "garantido",
    "garantida",
    "100%",
    "sem risco",
    "isento",
    "isenção",
    "não precisa ler",
    "nao precisa ler",
    "zero risco",
    "nunca falha",
)
PRESSURE = ("só hoje", "so hoje", "última chance", "ultima chance", "agora ou nunca", "urgente")
SENSITIVE_REQUEST = ("senha", "cartão de crédito", "cartao de credito", "cvv")


def _lines(transcript: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for raw in transcript.splitlines():
        match = _LINE_RE.match(raw.strip())
        if match:
            out.append((match.group(1), match.group(2)))
    return out


def _contains(text: str, needles: tuple[str, ...]) -> list[str]:
    low = text.lower()
    return [n for n in needles if n in low]


def _snippet(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0]


class FakeLLMClient:
    """Heuristic judge with the same interface as `AnthropicLLMClient`."""

    def __init__(self, *, model: str = "fake-judge-v1") -> None:
        self._model = model
        self.calls: int = 0

    @property
    def model(self) -> str:
        return self._model

    def parse_structured[T: BaseModel](
        self, *, system: str, user: str, output_format: type[T]
    ) -> LLMResult[T]:
        self.calls += 1
        if output_format is not CriterionGrade:
            raise LLMError(f"FakeLLMClient cannot produce {output_format.__name__}")
        criterion_match = _CRITERION_RE.search(user)
        transcript_match = _TRANSCRIPT_RE.search(user)
        if not criterion_match or not transcript_match:
            raise LLMError("prompt does not contain criterion/transcript markers")
        grade = self.grade(criterion_match.group(1), transcript_match.group(1))
        usage = LLMUsage(
            input_tokens=max(1, (len(system) + len(user)) // 4),
            output_tokens=max(1, len(grade.model_dump_json()) // 4),
        )
        return LLMResult(parsed=grade, usage=usage, model=self._model)  # type: ignore[arg-type]

    # -- heuristics ---------------------------------------------------------------------

    def grade(self, criterion_id: str, transcript: str) -> CriterionGrade:
        lines = _lines(transcript)
        agent = [t for r, t in lines if r == "agent"]
        customer = [t for r, t in lines if r == "customer"]
        handler = getattr(self, f"_grade_{criterion_id}", None)
        if handler is None:
            return CriterionGrade(
                score=3,
                rationale=f"Criterio {criterion_id} sem heuristica especifica; nota neutra.",
                evidence=[_snippet(agent[0])] if agent else [],
            )
        return handler(agent, customer)

    def _grade_opening(self, agent: list[str], customer: list[str]) -> CriterionGrade:
        if not agent:
            return CriterionGrade(score=0, rationale="Nenhuma mensagem do vendedor.", evidence=[])
        first = agent[0]
        greeting = _contains(first, GREETINGS)
        intro = _contains(first, INTRO)
        score = 1 if greeting else 0
        if intro:
            score += 2
        if greeting and intro and "?" in first:
            score = 5
        elif greeting and intro:
            score = 4
        return CriterionGrade(
            score=min(score, 5),
            rationale=(
                "Abertura com saudacao e apresentacao."
                if greeting and intro
                else "Abertura incompleta: falta saudacao ou apresentacao."
            ),
            evidence=[_snippet(first)],
        )

    def _grade_discovery(self, agent: list[str], customer: list[str]) -> CriterionGrade:
        questions = [t for t in agent if "?" in t]
        open_questions = [
            t for t in questions if _contains(t, ("como", "qual", "quais", "o que", "por que"))
        ]
        score = min(5, len(questions) + len(open_questions))
        if not questions:
            score = 0
        return CriterionGrade(
            score=score,
            rationale=(
                f"{len(questions)} pergunta(s) do vendedor, {len(open_questions)} aberta(s)."
                if questions
                else "O vendedor nao fez perguntas de descoberta."
            ),
            evidence=[_snippet(q) for q in questions[:2]]
            or ([_snippet(agent[0])] if agent else []),
        )

    def _grade_objection_handling(self, agent: list[str], customer: list[str]) -> CriterionGrade:
        objections = [t for t in customer if _contains(t, OBJECTIONS)]
        if not objections:
            return CriterionGrade(
                score=3,
                rationale="O lead nao levantou objecoes explicitas; nota neutra.",
                evidence=[_snippet(agent[0])] if agent else [],
            )
        acknowledging = [t for t in agent if _contains(t, ACKNOWLEDGE)]
        alternatives = [t for t in agent if _contains(t, ALTERNATIVES)]
        score = 1
        if acknowledging:
            score += 2
        if alternatives:
            score += 2
        evidence = [_snippet(objections[0])]
        evidence += [_snippet(t) for t in (acknowledging[:1] + alternatives[:1])]
        return CriterionGrade(
            score=min(score, 5),
            rationale=(
                "Objecao acolhida e respondida com alternativa."
                if acknowledging and alternatives
                else "Objecao tratada parcialmente."
            ),
            evidence=evidence[:3],
        )

    def _grade_next_steps(self, agent: list[str], customer: list[str]) -> CriterionGrade:
        tail = agent[-2:] if agent else []
        hits = [t for t in tail if _contains(t, NEXT_STEPS)]
        if not hits:
            return CriterionGrade(
                score=0 if not agent else 1,
                rationale="Conversa termina sem encaminhamento claro.",
                evidence=[_snippet(agent[-1])] if agent else [],
            )
        timed = [t for t in hits if _contains(t, TIME_HINT)]
        confirmed = bool(customer) and bool(
            _contains(customer[-1], ("combinado", "pode ser", "ok", "fechado", "perfeito", "sim"))
        )
        score = 3 + (1 if timed else 0) + (1 if confirmed else 0)
        return CriterionGrade(
            score=score,
            rationale=(
                "Encaminhamento definido com data e confirmacao."
                if score == 5
                else "Encaminhamento definido."
            ),
            evidence=[_snippet(hits[-1])],
        )

    def _grade_compliance(self, agent: list[str], customer: list[str]) -> CriterionGrade:
        promises = [t for t in agent if _contains(t, BAD_PROMISES)]
        pressure = [t for t in agent if _contains(t, PRESSURE)]
        sensitive = [t for t in agent if _contains(t, SENSITIVE_REQUEST)]
        if promises or sensitive:
            return CriterionGrade(
                score=0,
                rationale="Promessa indevida ou solicitacao de dado sensivel detectada.",
                evidence=[_snippet(t) for t in (promises + sensitive)[:2]],
            )
        if pressure:
            return CriterionGrade(
                score=1,
                rationale="Urgencia artificial detectada.",
                evidence=[_snippet(t) for t in pressure[:2]],
            )
        return CriterionGrade(
            score=5,
            rationale="Nenhuma promessa indevida ou pressao detectada.",
            evidence=[_snippet(agent[-1])] if agent else [],
        )
