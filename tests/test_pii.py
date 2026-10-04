import pytest

from grader.pii import redact_pii


@pytest.mark.parametrize(
    "text, token",
    [
        ("meu cpf é 000.000.000-00 ok", "CPF"),
        ("cpf 00000000000 sem pontos", "CPF"),
        ("liga no 11 90000-0000", "PHONE"),
        ("liga no (11) 90000-0000", "PHONE"),
        ("liga no +55 11 900000000", "PHONE"),
        ("fixo 3000-0000", "PHONE"),
        ("manda para ana.exemplo@example.com", "EMAIL"),
        ("cartão 0000 0000 0000 0000", "CARD"),
        ("cartão 0000-0000-0000-0000", "CARD"),
    ],
)
def test_single_entity(text: str, token: str):
    result = redact_pii(text)
    assert f"[{token}]" in result.text
    assert result.counts == {token: 1}
    assert not any(ch.isdigit() for ch in result.text)


def test_multiple_entities_and_counts():
    text = "CPF 000.000.000-00, tel 11 90000-0000, mail a@example.com e b@example.com"
    result = redact_pii(text)
    assert result.counts == {"CPF": 1, "PHONE": 1, "EMAIL": 2}
    assert result.total == 4
    assert result.text == "CPF [CPF], tel [PHONE], mail [EMAIL] e [EMAIL]"


def test_card_is_not_split_into_cpf_or_phone():
    result = redact_pii("0000 0000 0000 0000")
    assert result.text == "[CARD]"
    assert result.counts == {"CARD": 1}


def test_no_false_positives_on_business_numbers():
    text = "Fica em torno de 12 mil por mês, consumo de 800 kWh, desconto de 30% em 2026."
    result = redact_pii(text)
    assert result.text == text
    assert result.counts == {}
