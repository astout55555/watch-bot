"""Ingest current-Congress bills from congress.gov into the local vector index.

Uses the bulk list endpoints -- ``/v3/bill/{congress}`` for bills and
``/v3/summaries/{congress}/{type}`` for CRS summaries -- so a full-Congress
ingest is on the order of a hundred requests, not one per bill. Bills are
keyed by GovQL's canonical bill_id (e.g. ``hr1181-119``) so the index joins
cleanly against GovQL vote data.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import psycopg

from watchbot import db, embeddings
from watchbot.billref import BILL_TYPES, canonical_bill_id, display_name, parse_bill_id
from watchbot.config import settings

API_BASE = "https://api.congress.gov/v3"
PAGE_SIZE = 250
# text-embedding-3-small caps input at 8191 tokens; ~20k chars is a safe bound.
MAX_EMBED_CHARS = 20_000
EMBED_BATCH = 100

_TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class BillRecord:
    bill_id: str
    bill_type: str
    number: int
    congress: int
    title: str
    summary: str | None
    latest_action: str | None
    _summary_date: str = ""


def plain_text(html_text: str) -> str:
    """Strip HTML tags and collapse whitespace from a CRS summary."""
    return re.sub(r"\s+", " ", html.unescape(_TAG_RE.sub(" ", html_text))).strip()


def parse_bill_item(item: dict) -> BillRecord:
    congress = int(item["congress"])
    number = int(item["number"])
    action = item.get("latestAction") or {}
    latest_action = None
    if action.get("text"):
        latest_action = f"{action.get('actionDate', '?')}: {action['text']}"
    return BillRecord(
        bill_id=canonical_bill_id(item["type"], number, congress),
        bill_type=item["type"].lower(),
        number=number,
        congress=congress,
        title=item.get("title") or "(untitled)",
        summary=None,
        latest_action=latest_action,
    )


def parse_summary_item(item: dict) -> tuple[str, str, str]:
    """Return (bill_id, action_date, plain-text summary)."""
    bill = item["bill"]
    bill_id = canonical_bill_id(bill["type"], int(bill["number"]), int(bill["congress"]))
    return bill_id, item.get("actionDate", ""), plain_text(item.get("text", ""))


def merge_summaries(
    bills: dict[str, BillRecord], summaries: list[tuple[str, str, str]]
) -> None:
    """Attach each bill's most recent summary; ignore summaries for unknown bills."""
    for bill_id, action_date, text in summaries:
        record = bills.get(bill_id)
        if record is not None and action_date >= record._summary_date:
            record.summary = text
            record._summary_date = action_date


def embedding_text(record: BillRecord) -> str:
    ref = parse_bill_id(record.bill_id)
    text = f"{display_name(ref)} — {record.title}"
    if record.summary:
        text += f"\n\n{record.summary}"
    return text[:MAX_EMBED_CHARS]


class CongressGovClient:
    def __init__(self, api_key: str, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=30)
        self._api_key = api_key

    def paged(self, path: str, max_items: int | None = None) -> Iterator[dict]:
        """Yield the per-item payloads of a paginated list endpoint."""
        offset = 0
        yielded = 0
        while True:
            response = self._request(path, offset)
            data = response.json()
            # The list key varies by endpoint ("bills", "summaries").
            key = next(k for k in data if k not in ("pagination", "request"))
            items = data[key]
            for item in items:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            if data.get("pagination", {}).get("next") is None or not items:
                return
            offset += PAGE_SIZE

    def _request(self, path: str, offset: int) -> httpx.Response:
        for _ in range(4):
            response = self._client.get(
                f"{API_BASE}{path}",
                params={
                    "api_key": self._api_key,
                    "format": "json",
                    "limit": PAGE_SIZE,
                    "offset": offset,
                },
            )
            if response.status_code == 429:
                wait = int(response.headers.get("retry-after", 30))
                print(f"  rate limited; sleeping {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        raise RuntimeError(f"Still rate-limited after retries: {path}")


def fetch_bills(
    client: CongressGovClient, congress: int, max_items: int | None = None
) -> dict[str, BillRecord]:
    bills: dict[str, BillRecord] = {}
    for item in client.paged(f"/bill/{congress}", max_items=max_items):
        record = parse_bill_item(item)
        bills[record.bill_id] = record
    return bills


def fetch_summaries(client: CongressGovClient, congress: int) -> list[tuple[str, str, str]]:
    summaries = []
    for bill_type in BILL_TYPES:
        for item in client.paged(f"/summaries/{congress}/{bill_type}"):
            summaries.append(parse_summary_item(item))
    return summaries


def upsert_bills(conn: psycopg.Connection, records: list[BillRecord]) -> None:
    if not records:
        return
    vectors = embeddings.embed_texts([embedding_text(r) for r in records])
    with conn.cursor() as cur:
        for record, vector in zip(records, vectors, strict=True):
            cur.execute(
                """
                INSERT INTO bills
                    (bill_id, bill_type, bill_number, congress, title, summary,
                     latest_action, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (bill_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    latest_action = EXCLUDED.latest_action,
                    embedding = EXCLUDED.embedding
                """,
                (
                    record.bill_id,
                    record.bill_type,
                    record.number,
                    record.congress,
                    record.title,
                    record.summary,
                    record.latest_action,
                    vector,
                ),
            )
    conn.commit()


def ingest(
    conn: psycopg.Connection,
    client: CongressGovClient,
    congress: int,
    max_bills: int | None = None,
    refresh: bool = False,
) -> int:
    print(f"Fetching bills for the {congress}th Congress...")
    bills = fetch_bills(client, congress, max_items=max_bills)
    print(f"  {len(bills)} bills fetched.")

    print("Fetching CRS summaries...")
    merge_summaries(bills, fetch_summaries(client, congress))
    with_summary = sum(1 for b in bills.values() if b.summary)
    print(f"  {with_summary} bills have summaries.")

    if not refresh:
        existing = {
            row[0]
            for row in conn.execute(
                "SELECT bill_id FROM bills WHERE congress = %s", (congress,)
            )
        }
        bills = {k: v for k, v in bills.items() if k not in existing}
        print(f"  {len(bills)} bills are new (use --refresh to re-embed everything).")

    records = list(bills.values())
    for start in range(0, len(records), EMBED_BATCH):
        batch = records[start : start + EMBED_BATCH]
        upsert_bills(conn, batch)
        print(f"  embedded and stored {min(start + EMBED_BATCH, len(records))}/{len(records)}")
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest congress.gov bills into WatchBot.")
    parser.add_argument("--limit", type=int, default=None, help="Max bills to fetch (for testing)")
    parser.add_argument(
        "--refresh", action="store_true", help="Re-embed bills already in the database"
    )
    args = parser.parse_args()

    config = settings()
    if not config.congress_gov_api_key:
        sys.exit("CONGRESS_GOV_API_KEY is not set (get one at https://api.congress.gov/sign-up/)")

    client = CongressGovClient(config.congress_gov_api_key)
    with db.connect() as conn:
        db.setup(conn)
        stored = ingest(conn, client, config.congress, max_bills=args.limit, refresh=args.refresh)
    print(f"Done: {stored} bills stored.")


if __name__ == "__main__":
    main()
