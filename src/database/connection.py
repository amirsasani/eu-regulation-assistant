import os

import psycopg
from psycopg.rows import dict_row


DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/eu_regulations"


def get_connection():
    database_url = os.getenv(
        "DATABASE_URL",
        DEFAULT_DATABASE_URL,
    )

    return psycopg.connect(
        database_url,
        row_factory=dict_row,
    )