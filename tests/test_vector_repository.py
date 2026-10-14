import numpy as np

from src.database import vector_repository


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def execute(self, query, params):
        self.query = query
        self.params = params

    def fetchall(self):
        return [
            {
                "chunk_id": "ai-act-it-article-9-paragraph-1",
                "document_id": "32024R1689",
                "language": "it",
                "article_number": "9",
                "article_title": "Sistema di gestione dei rischi",
                "paragraph_number": "1",
                "text": "È istituito un sistema di gestione dei rischi.",
                "source_url": "https://eur-lex.europa.eu/",
                "similarity": 0.9,
            }
        ]


class FakeConnection:
    def __init__(self):
        self.fake_cursor = FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def cursor(self):
        return self.fake_cursor


def test_search_similar_filters_by_language(monkeypatch):
    connection = FakeConnection()
    monkeypatch.setattr(
        vector_repository,
        "get_connection",
        lambda: connection,
    )
    monkeypatch.setattr(
        vector_repository,
        "register_vector",
        lambda _: None,
    )

    results = vector_repository.search_similar(
        query_embedding=np.zeros(768, dtype=np.float32),
        top_k=5,
        language="it",
    )

    assert "WHERE language = %s" in connection.fake_cursor.query
    assert connection.fake_cursor.params[1] == "it"
    assert results[0]["language"] == "it"
