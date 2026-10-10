import re


CITATION_PATTERN = re.compile(r"\[Article\s+(\d+),\s*paragraph\s+(\d+)\]", re.IGNORECASE)

REFUSAL_PHRASES = (
    "not enough information",
    "insufficient information",
    "does not contain",
    "do not contain",
    "does not specify",
    "do not specify",
    "cannot determine",
)


def extract_citations(answer: str) -> list[dict]:
    return [
        {"article": article, "paragraph": paragraph}
        for article, paragraph
        in CITATION_PATTERN.findall(answer)
    ]


def evaluate_citations(answer: str, sources: list[dict], answerable: bool) -> dict:
    citations = extract_citations(answer)

    available_citations = {
        (str(source["article_number"]), str(source["paragraph_number"]))
        for source in sources
        if source.get("paragraph_number") is not None
    }

    valid = [
        citation
        for citation in citations
        if (citation["article"], citation["paragraph"]) in available_citations
    ]

    invalid = [
        citation
        for citation in citations
        if (citation["article"], citation["paragraph"]) not in available_citations
    ]

    if citations:
        precision = len(valid) / len(citations)
    else:
        precision = 1.0 if not answerable else 0.0

    return {
        "citations": citations,
        "valid_citations": valid,
        "invalid_citations": invalid,
        "citation_precision": round(precision, 3),
        "has_valid_citation": bool(valid),
    }


def calculate_expected_source_recall(expected_points: list[dict], sources: list[dict]) -> float:
    expected_chunks = {point["chunk_id"] for point in expected_points}
    retrieved_chunks = {source["chunk_id"] for source in sources}

    if not expected_chunks:
        return 1.0

    recalled_chunks = (expected_chunks & retrieved_chunks)

    return round(len(recalled_chunks) / len(expected_chunks), 3)


def contains_refusal(answer: str) -> bool:
    normalized_answer = answer.lower()

    return any(phrase in normalized_answer for phrase in REFUSAL_PHRASES)


def evaluate_answer(answer: str, sources: list[dict], evaluation_item: dict) -> dict:
    answerable = evaluation_item["answerable"]
    refused = contains_refusal(answer)

    citation_results = evaluate_citations(answer=answer, sources=sources, answerable=answerable)

    return {
        **citation_results,
        "expected_source_recall": (
            calculate_expected_source_recall(
                expected_points=(evaluation_item["expected_points"]),
                sources=sources,
            )
        ),
        "refused": refused,
        "refusal_correct": (refused if not answerable else not refused),
    }