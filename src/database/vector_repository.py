import numpy as np
from pgvector.psycopg import register_vector

from src.database.connection import get_connection


UPSERT_SQL = """
INSERT INTO regulation_chunks (
    chunk_id,
    document_id,
    language,
    article_number,
    article_title,
    paragraph_number,
    text,
    source_url,
    embedding
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s
)
ON CONFLICT (chunk_id)
DO UPDATE SET
    document_id = EXCLUDED.document_id,
    language = EXCLUDED.language,
    article_number = EXCLUDED.article_number,
    article_title = EXCLUDED.article_title,
    paragraph_number = EXCLUDED.paragraph_number,
    text = EXCLUDED.text,
    source_url = EXCLUDED.source_url,
    embedding = EXCLUDED.embedding,
    updated_at = NOW();
"""

SEARCH_SQL = """
SELECT
    chunk_id,
    document_id,
    language,
    article_number,
    article_title,
    paragraph_number,
    text,
    source_url,
    1 - (embedding <=> %s) AS similarity
FROM regulation_chunks
ORDER BY embedding <=> %s
LIMIT %s;
"""

def upsert_documents(documents: list[dict], embeddings: np.ndarray, default_document_id: str) -> int:
    if len(documents) != len(embeddings):
        raise ValueError("Document and embedding counts do not match.")

    rows = []

    for document, embedding in zip(documents, embeddings):
        rows.append((
            document["chunk_id"],
            document.get("document_id", default_document_id),
            document.get("language", "en"),
            str(document["article_number"]),
            document.get("article_title"),
            document.get("paragraph_number"),
            document["text"],
            document.get("source_url"),
            np.asarray(embedding, dtype=np.float32),
        ))

    with get_connection() as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.executemany(
                UPSERT_SQL,
                rows,
            )

    return len(rows)


def search_similar(query_embedding: np.ndarray, top_k: int = 5):
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    with get_connection() as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                SEARCH_SQL,
                (
                    query_embedding,
                    query_embedding,
                    top_k,
                ),
            )

            rows = cursor.fetchall()

    return [
        {**dict(row), "similarity": float(row["similarity"])}
        for row in rows
    ]