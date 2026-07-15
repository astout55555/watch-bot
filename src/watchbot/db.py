"""Database connection and schema setup."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from watchbot.config import EMBEDDING_DIMENSIONS, settings

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS bills (
    id BIGSERIAL PRIMARY KEY,
    bill_id TEXT NOT NULL UNIQUE,       -- canonical GovQL form, e.g. 'hr1181-119'
    bill_type TEXT NOT NULL,
    bill_number INTEGER NOT NULL,
    congress SMALLINT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    sponsor TEXT,
    latest_action TEXT,
    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bills_congress ON bills (congress);
"""


def connect(database_url: str | None = None) -> psycopg.Connection:
    conn = psycopg.connect(database_url or settings().database_url)
    # register_vector needs the extension to exist; tolerate a fresh database
    # so setup() can run against it with this same helper.
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    register_vector(conn)
    return conn


def setup(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()


def main() -> None:
    with connect() as conn:
        setup(conn)
    print("Database schema ready.")


if __name__ == "__main__":
    main()
