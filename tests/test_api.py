import importlib

import pytest
from fastapi.testclient import TestClient


api_module = importlib.import_module("src.api.app")


class FakeRagService:
    def __init__(self):
        pass

    def ask(
        self,
        question: str,
        top_k: int = 5,
        candidate_k: int = 10,
        llm_model: str | None = None,
    ):
        if candidate_k < top_k:
            raise ValueError(
                "candidate_k must be greater than or equal to top_k."
            )

        return (
            {
                "answer": (
                    "A risk management system is required "
                    "[Article 9, paragraph 1]."
                ),
                "model": "qwen3.5:4b",
            },
            [
                {
                    "chunk_id": "ai-act-en-article-9-paragraph-1",
                    "article_number": "9",
                    "article_title": "Risk management system",
                    "paragraph_number": "1",
                    "text": "A risk management system shall be established.",
                    "source_url": "https://eur-lex.europa.eu/",
                    "reranker_score": 0.99,
                }
            ],
        )


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query: str):
        self.query = query

    def fetchone(self):
        return {"count": 519}


class FakeConnection:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self):
        return FakeCursor()


class FakeOllamaResponse:
    def raise_for_status(self):
        pass

    def json(self):
        return {"models": [{"name": "qwen3.5:4b"}]}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "RagService", FakeRagService)

    with TestClient(api_module.app) as test_client:
        yield test_client


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_returns_dependency_status(client, monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setattr(
        api_module,
        "get_connection",
        lambda: FakeConnection(),
    )
    monkeypatch.setattr(
        api_module.requests,
        "get",
        lambda *args, **kwargs: FakeOllamaResponse(),
    )

    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "services": {
            "postgres": {
                "status": "ok",
                "chunks": 519,
            },
            "ollama": {
                "status": "ok",
                "model": "qwen3.5:4b",
            },
        },
    }


def test_readiness_returns_503_when_postgres_is_unavailable(
    client,
    monkeypatch,
):
    def unavailable_database():
        raise ConnectionError("PostgreSQL is unavailable")

    monkeypatch.setenv("OLLAMA_MODEL", "qwen3.5:4b")
    monkeypatch.setattr(
        api_module,
        "get_connection",
        unavailable_database,
    )
    monkeypatch.setattr(
        api_module.requests,
        "get",
        lambda *args, **kwargs: FakeOllamaResponse(),
    )

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
    assert response.json()["services"]["postgres"] == {
        "status": "unavailable",
        "chunks": None,
    }


def test_ask_returns_answer_and_sources(client):
    response = client.post(
        "/ask",
        json={
            "question": (
                "What risk management obligations apply to "
                "high-risk AI systems?"
            ),
            "top_k": 5,
            "candidate_k": 10,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["model"] == "qwen3.5:4b"
    assert payload["sources"][0]["article_number"] == "9"
    assert payload["sources"][0]["paragraph_number"] == "1"


def test_ask_rejects_invalid_retrieval_limits(client):
    response = client.post(
        "/ask",
        json={
            "question": "What does Article 9 require?",
            "top_k": 10,
            "candidate_k": 5,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "candidate_k must be greater than or equal to top_k."
    )


def test_ask_rejects_empty_question(client):
    response = client.post(
        "/ask",
        json={
            "question": "",
            "top_k": 5,
            "candidate_k": 10,
        },
    )

    assert response.status_code == 422
