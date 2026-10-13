from src.evaluation.generation import contains_refusal


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
