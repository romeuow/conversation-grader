from grader.evidence import normalize, verify_evidence

SOURCE = "agent: Olá, bom dia! Meu nome é Ana Exemplo.\ncustomer: Bom dia, Ana."


def test_normalize_strips_accents_case_punctuation_whitespace():
    assert normalize("  Olá,   BOM dia!  ") == "ola bom dia"
    assert normalize("[CPF]") == "[cpf]"


def test_verbatim_evidence_is_verified():
    assert verify_evidence(["Meu nome é Ana Exemplo"], SOURCE)


def test_evidence_tolerates_case_and_accent_drift():
    assert verify_evidence(["meu nome e ana exemplo"], SOURCE)


def test_paraphrased_evidence_fails():
    assert not verify_evidence(["A vendedora se apresentou"], SOURCE)


def test_all_excerpts_must_match():
    assert not verify_evidence(["Bom dia, Ana", "frase inventada"], SOURCE)


def test_empty_or_too_short_evidence_fails():
    assert not verify_evidence([], SOURCE)
    assert not verify_evidence(["a"], SOURCE)
