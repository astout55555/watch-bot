"""Semantic search over the local bills index."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from watchbot import embeddings


@dataclass(frozen=True)
class BillHit:
    bill_id: str
    title: str
    summary: str | None
    congress: int
    latest_action: str | None
    similarity: float


def search_bills(
    conn: psycopg.Connection,
    query: str,
    congress: int,
    limit: int = 8,
    query_embedding: list[float] | None = None,
) -> list[BillHit]:
    """Return the bills of one Congress most semantically similar to `query`.

    `query_embedding` lets tests (or batch callers) skip the OpenAI call.
    """
    if query_embedding is None:
        query_embedding = embeddings.embed_query(query)

    rows = conn.execute(
        """
        SELECT bill_id, title, summary, congress, latest_action,
               1 - (embedding <=> %s::vector) AS similarity
        FROM bills
        WHERE congress = %s
        ORDER BY similarity DESC
        LIMIT %s
        """,
        (query_embedding, congress, limit),
    ).fetchall()

    return [BillHit(*row) for row in rows]
