from src.database.connection import get_connection


EMBEDDING_DIMENSION = 768


CREATE_EXTENSION_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;
"""


CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS regulation_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    article_number TEXT NOT NULL,
    article_title TEXT,
    paragraph_number TEXT,
    text TEXT NOT NULL,
    source_url TEXT,
    embedding vector({EMBEDDING_DIMENSION}) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def initialize_schema() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(CREATE_EXTENSION_SQL)
            cursor.execute(CREATE_TABLE_SQL)

    print("Database schema initialized.")