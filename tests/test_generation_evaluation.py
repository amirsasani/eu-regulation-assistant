from src.evaluation.generation import contains_refusal, extract_citations


def test_detects_refusal_in_opening_paragraph():
    answer = (
        "Based on the supplied excerpts, there is no information about "
        "the required encryption algorithm.\n\nThe excerpts discuss cybersecurity."
    )

    assert contains_refusal(answer) is True


def test_ignores_late_caveat_after_substantive_answer():
    answer = (
        "Article 6 explains the classification criteria and the answer follows "
        "from those criteria. [Article 6, paragraph 1]\n\n"
        "A later caveat says the excerpts do not contain every Annex detail."
    )

    assert contains_refusal(answer) is False


def test_detects_italian_refusal_in_opening_paragraph():
    answer = (
        "Gli estratti forniti non contengono informazioni sufficienti "
        "sull'algoritmo di cifratura richiesto.\n\n"
        "Gli estratti riguardano invece la cibersicurezza."
    )

    assert contains_refusal(answer) is True


def test_extracts_full_and_abbreviated_italian_citations():
    answer = (
        "Il sistema richiede misure continue [Articolo 9, paragrafo 2] "
        "e prove adeguate [Art. 9, par. 6]."
    )

    assert extract_citations(answer) == [
        {"article": "9", "paragraph": "2"},
        {"article": "9", "paragraph": "6"},
    ]
