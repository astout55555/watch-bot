"""Integration tests for schema setup and vector search.

These run against the pgvector container from compose.yml (port 5433) and are
skipped automatically when it isn't reachable. No OpenAI calls are made --
embeddings are stubbed with fixed vectors.
"""

import math

import psycopg
import pytest

from watchbot import db
from watchbot.config import EMBEDDING_DIMENSIONS, settings
from watchbot.search import search_bills


def _unit_vector(hot_index: int) -> list[float]:
    vec = [0.0] * EMBEDDING_DIMENSIONS
    vec[hot_index] = 1.0
    return vec


def _blend(a: list[float], b: list[float], weight: float) -> list[float]:
    raw = [weight * x + (1 - weight) * y for x, y in zip(a, b, strict=True)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


@pytest.fixture(scope="module")
def conn():
    try:
        connection = db.connect(settings().database_url)
    except psycopg.OperationalError:
        pytest.skip("Postgres (compose.yml container) is not running")
    db.setup(connection)
    yield connection
    connection.close()


@pytest.fixture()
def seeded(conn):
    conn.execute("TRUNCATE bills")
    privacy = _unit_vector(0)
    daylight = _unit_vector(1)
    rows = [
        ("hr1181-119", "hr", 1181, 119, "Protecting Privacy in Purchases Act", privacy),
        ("hr139-119", "hr", 139, 119, "Sunshine Protection Act", daylight),
    ]
    for bill_id, bill_type, number, congress, title, embedding in rows:
        conn.execute(
            """
            INSERT INTO bills (bill_id, bill_type, bill_number, congress, title, embedding)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (bill_id, bill_type, number, congress, title, embedding),
        )
    conn.commit()
    return {"privacy": privacy, "daylight": daylight}


class TestSearchBills:
    def test_ranks_closest_bill_first(self, conn, seeded):
        near_privacy = _blend(seeded["privacy"], seeded["daylight"], 0.9)
        hits = search_bills(conn, "purchase privacy", query_embedding=near_privacy)
        assert [h.bill_id for h in hits] == ["hr1181-119", "hr139-119"]
        assert hits[0].similarity > hits[1].similarity
        assert hits[0].title == "Protecting Privacy in Purchases Act"

    def test_respects_limit(self, conn, seeded):
        hits = search_bills(conn, "anything", limit=1, query_embedding=seeded["daylight"])
        assert len(hits) == 1
        assert hits[0].bill_id == "hr139-119"

    def test_setup_is_idempotent(self, conn):
        db.setup(conn)
        db.setup(conn)
