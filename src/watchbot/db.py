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
    latest_action TEXT,
    embedding vector({EMBEDDING_DIMENSIONS}) NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bills_congress ON bills (congress);
CREATE INDEX IF NOT EXISTS idx_bills_embedding
    ON bills USING hnsw (embedding vector_cosine_ops);

-- One row per Congress: when its bills/summaries were last fetched, so
-- ingest runs can pull only what changed since.
CREATE TABLE IF NOT EXISTS ingest_runs (
    congress SMALLINT PRIMARY KEY,
    last_fetched_at TIMESTAMPTZ NOT NULL
);
"""


def connect(database_url: str | None = None) -> psycopg.Connection:
    # autocommit keeps the long-lived REPL connection out of "idle in
    # transaction" between turns (read queries would otherwise hold one open).
    conn = psycopg.connect(database_url or settings().database_url, autocommit=True)
    # register_vector needs the extension to exist; create it here so this
    # helper also works against a fresh database before setup() runs.
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def setup(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def main() -> None:
    with connect() as conn:
        setup(conn)
    print("Database schema ready.")


if __name__ == "__main__":
    main()
