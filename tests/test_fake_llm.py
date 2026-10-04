import pytest
from pydantic import BaseModel

from grader.llm.base import LLMError
from grader.llm.fake import FakeLLMClient
from grader.models import CriterionGrade
from grader.prompts import SYSTEM_PROMPT, build_user_prompt


def test_is_deterministic(rubric, by_id):
    client = FakeLLMClient()
    transcript = by_id["conv-001"].transcript()
    prompt = build_user_prompt(rubric, rubric.get("discovery"), transcript)
    first = client.parse_structured(system=SYSTEM_PROMPT, user=prompt, output_format=CriterionGrade)
    second = client.parse_structured(
        system=SYSTEM_PROMPT, user=prompt, output_format=CriterionGrade
    )
    assert first.parsed == second.parsed
    assert first.usage == second.usage
    assert first.usage.input_tokens > 0 and first.usage.output_tokens > 0
    assert client.calls == 2
    assert client.model == "fake-judge-v1"


def test_rejects_unknown_output_format():
    class Other(BaseModel):
        x: int

    with pytest.raises(LLMError, match="cannot produce Other"):
        FakeLLMClient().parse_structured(system="s", user="u", output_format=Other)


def test_requires_prompt_markers():
    with pytest.raises(LLMError, match="markers"):
        FakeLLMClient().parse_structured(
            system="s", user="no markers here", output_format=CriterionGrade
        )


def test_unknown_criterion_gets_neutral_score():
    grade = FakeLLMClient().grade("made_up", "agent: Olá\ncustomer: oi")
    assert grade.score == 3
    assert grade.evidence == ["Olá"]


@pytest.mark.parametrize(
    "transcript, expected",
    [
        ("agent: Olá, bom dia! Meu nome é Ana. Tudo bem?\ncustomer: oi", 5),
        ("agent: Olá, bom dia! Meu nome é Ana.\ncustomer: oi", 4),
        ("agent: Olá!\ncustomer: oi", 1),
        ("agent: Quer contratar?\ncustomer: não", 0),
        ("customer: alguém aí?", 0),
    ],
)
def test_opening_heuristic(transcript: str, expected: int):
    assert FakeLLMClient().grade("opening", transcript).score == expected


def test_discovery_counts_questions():
    client = FakeLLMClient()
    assert client.grade("discovery", "agent: Oferta boa.\ncustomer: ok").score == 0
    assert client.grade("discovery", "agent: Quer?\ncustomer: ok").score == 1
    assert client.grade("discovery", "agent: Como é hoje? Qual o valor?\ncustomer: ok").score == 2
    many = "\n".join(f"agent: Como vai {i}?" for i in range(5))
    assert client.grade("discovery", many).score == 5


def test_objection_handling_levels():
    client = FakeLLMClient()
    assert client.grade("objection_handling", "agent: Oi\ncustomer: legal").score == 3
    assert client.grade("objection_handling", "agent: Oi\ncustomer: Acho caro.").score == 1
    ack = "customer: Acho caro.\nagent: Entendo você."
    assert client.grade("objection_handling", ack).score == 3
    full = ack + "\nagent: Podemos ver uma alternativa sem compromisso."
    grade = client.grade("objection_handling", full)
    assert grade.score == 5
    assert len(grade.evidence) == 3


def test_next_steps_levels():
    client = FakeLLMClient()
    assert client.grade("next_steps", "customer: oi").score == 0
    assert client.grade("next_steps", "agent: tchau\ncustomer: tchau").score == 1
    assert client.grade("next_steps", "agent: Envio a proposta.\ncustomer: hmm").score == 3
    timed = "agent: Agendo para quarta às 10h.\ncustomer: hmm"
    assert client.grade("next_steps", timed).score == 4
    confirmed = "agent: Agendo para quarta às 10h.\ncustomer: Combinado"
    assert client.grade("next_steps", confirmed).score == 5


def test_compliance_levels():
    client = FakeLLMClient()
    assert client.grade("compliance", "agent: Eu garanto economia.\ncustomer: ok").score == 0
    assert client.grade("compliance", "agent: Me passa sua senha.\ncustomer: não").score == 0
    assert client.grade("compliance", "agent: É só hoje!\ncustomer: hmm").score == 1
    assert client.grade("compliance", "agent: Posso enviar a proposta?\ncustomer: sim").score == 5
    assert client.grade("compliance", "customer: oi").score == 5
