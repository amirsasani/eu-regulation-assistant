from src.generation.prompt import build_messages


def test_build_messages_requests_an_italian_answer():
    documents = [
        {
            "chunk_id": "ai-act-en-article-9-paragraph-1",
            "article_number": "9",
            "article_title": "Risk management system",
            "paragraph_number": "1",
            "text": "A risk management system shall be established.",
        }
    ]

    messages = build_messages(
        question="Quali obblighi di gestione del rischio si applicano?",
        documents=documents,
        language="it",
    )

    assert "Answer in Italian." in messages[-1]["content"]
    assert "Keep article and paragraph citations" in messages[-1]["content"]
