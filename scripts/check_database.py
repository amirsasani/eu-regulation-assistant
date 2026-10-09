from src.database.connection import get_connection


def main() -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT
                    current_database() AS database_name,
                    current_user AS database_user;
            """)

            database_info = cursor.fetchone()

            cursor.execute("""
                SELECT extversion
                FROM pg_extension
                WHERE extname = 'vector';
            """)

            vector_info = cursor.fetchone()

    print(f"Database: {database_info['database_name']}")
    print(f"User: {database_info['database_user']}")
    print(f"pgvector version: {vector_info['extversion']}")


if __name__ == "__main__":
    main()