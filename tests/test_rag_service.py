from src.services import rag_service as rag_service_module


def test_ask_passes_italian_to_retrieval_and_prompt(monkeypatch):
    service = object.__new__(rag_service_module.RagService)
    italian_retriever = object()
    service.bm25_retrievers = {"it": italian_retriever}
    service.embedding_model = object()
    service.reranker_model = object()

    source = {
        "chunk_id": "ai-act-en-article-9-paragraph-1",
        "article_number": "9",
        "article_title": "Risk management system",
        "paragraph_number": "1",
        "text": "A risk management system shall be established.",
    }
    observed = {}

    def fake_search_hybrid(**kwargs):
        observed["retrieval_language"] = kwargs["language"]
        observed["bm25_retriever"] = kwargs["bm25_retriever"]
        return [source]

    def fake_build_messages(question, documents, language):
        observed["prompt_language"] = language
        return [{"role": "user", "content": question}]

    def fake_generate_answer(messages, model):
        return {
            "answer": "È necessario un sistema di gestione dei rischi.",
            "model": "test-model",
        }

    monkeypatch.setattr(
        rag_service_module,
        "search_hybrid",
        fake_search_hybrid,
    )
    monkeypatch.setattr(
        rag_service_module,
        "build_messages",
        fake_build_messages,
    )
    monkeypatch.setattr(
        rag_service_module,
        "generate_answer",
        fake_generate_answer,
    )

    generation, sources = service.ask(
        question="Quali obblighi di gestione del rischio si applicano?",
        language="it",
    )

    assert observed["retrieval_language"] == "it"
    assert observed["prompt_language"] == "it"
    assert observed["bm25_retriever"] is italian_retriever
    assert generation["model"] == "test-model"
    assert sources == [source]


def test_ask_rejects_language_without_an_indexed_corpus():
    service = object.__new__(rag_service_module.RagService)
    service.bm25_retrievers = {"en": object()}

    try:
        service.ask(question="Quali obblighi si applicano?", language="it")
    except ValueError as error:
        assert str(error) == "No indexed corpus is available for language: it"
    else:
        raise AssertionError("Expected unavailable Italian corpus to be rejected")
