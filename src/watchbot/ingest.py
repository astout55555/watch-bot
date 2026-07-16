"""Ingest current-Congress bills from congress.gov into the local vector index.

Uses the bulk list endpoints -- ``/v3/bill/{congress}`` for bills and
``/v3/summaries/{congress}/{type}`` for CRS summaries -- so a full-Congress
ingest is on the order of a hundred requests, not one per bill. Each run
records a high-water mark in ``ingest_runs``; later runs pass it as
``fromDateTime`` so only bills and summaries changed since the last run are
fetched and re-embedded. Bills are keyed by GovQL's canonical bill_id
(e.g. ``hr1181-119``) so the index joins cleanly against GovQL vote data.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
# Re-fetch a little history before the stored high-water mark; upserts are
# idempotent, so the overlap only guards against clock skew with the API.
WATERMARK_OVERLAP = timedelta(days=1)

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


def parse_summary_item(item: dict) -> BillRecord:
    """Build a BillRecord carrying this summary, from the summary's own bill stub.

    Summaries can change without the parent bill appearing in an incremental
    bills fetch, so each summary carries enough to stand alone as an upsert.
    """
    bill = item["bill"]
    congress = int(bill["congress"])
    number = int(bill["number"])
    return BillRecord(
        bill_id=canonical_bill_id(bill["type"], number, congress),
        bill_type=bill["type"].lower(),
        number=number,
        congress=congress,
        title=bill.get("title") or "(untitled)",
        summary=plain_text(item.get("text", "")),
        latest_action=None,
        _summary_date=item.get("actionDate") or "",
    )


def merge_summaries(bills: dict[str, BillRecord], summaries: list[BillRecord]) -> None:
    """Fold summary records into the bills dict, keeping each bill's newest summary.

    Summaries for bills missing from the fetch are added as standalone records
    (a summary-only change on an incremental run still gets upserted).
    """
    for summary_record in summaries:
        record = bills.setdefault(summary_record.bill_id, summary_record)
        if summary_record._summary_date >= record._summary_date:
            record.summary = summary_record.summary
            record._summary_date = summary_record._summary_date


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

    def paged(
        self,
        path: str,
        key: str,
        max_items: int | None = None,
        params: dict | None = None,
    ) -> Iterator[dict]:
        """Yield the per-item payloads of a paginated list endpoint.

        `key` names the response's list field ("bills", "summaries") so an
        unexpected response shape fails loudly instead of guessing.
        """
        offset = 0
        yielded = 0
        while True:
            response = self._request(path, offset, params or {})
            data = response.json()
            if key not in data:
                raise RuntimeError(f"Unexpected response from {path}: keys {sorted(data)}")
            items = data[key]
            for item in items:
                yield item
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            if data.get("pagination", {}).get("next") is None or not items:
                return
            offset += PAGE_SIZE

    def _request(self, path: str, offset: int, params: dict) -> httpx.Response:
        for _ in range(4):
            response = self._client.get(
                f"{API_BASE}{path}",
                # Fixed keys come last so callers can never override pagination.
                params={
                    **params,
                    "api_key": self._api_key,
                    "format": "json",
                    "limit": PAGE_SIZE,
                    "offset": offset,
                },
            )
            if response.status_code == 429:
                wait = _retry_after_seconds(response.headers.get("retry-after"))
                print(f"  rate limited; sleeping {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response
        raise RuntimeError(f"Still rate-limited after retries: {path}")


def _retry_after_seconds(header: str | None, default: int = 30) -> int:
    """Retry-After may be delay-seconds or an HTTP-date; fall back on anything odd."""
    try:
        return int(header) if header else default
    except ValueError:
        return default


def fetch_bills(
    client: CongressGovClient,
    congress: int,
    since: str,
    max_items: int | None = None,
) -> dict[str, BillRecord]:
    bills: dict[str, BillRecord] = {}
    params = {"fromDateTime": since}
    for item in client.paged(f"/bill/{congress}", key="bills", max_items=max_items, params=params):
        record = parse_bill_item(item)
        bills[record.bill_id] = record
    return bills


def congress_start_year(congress: int) -> int:
    """First calendar year of a Congress (the 1st convened in 1789)."""
    return 1789 + (congress - 1) * 2


def fetch_summaries(
    client: CongressGovClient,
    congress: int,
    since: str,
    max_items: int | None = None,
) -> list[BillRecord]:
    # The summaries endpoint applies a narrow recent-updates window unless
    # fromDateTime is explicit, so always pass one.
    params = {"fromDateTime": since}
    summaries = []
    for bill_type in BILL_TYPES:
        path = f"/summaries/{congress}/{bill_type}"
        for item in client.paged(path, key="summaries", max_items=max_items, params=params):
            summaries.append(parse_summary_item(item))
    return summaries


def fetch_window_start(conn: psycopg.Connection, congress: int, refresh: bool) -> str:
    """fromDateTime for this run: just before the last run, or the Congress start."""
    row = None
    if not refresh:
        row = conn.execute(
            "SELECT last_fetched_at FROM ingest_runs WHERE congress = %s", (congress,)
        ).fetchone()
    if row is None:
        return f"{congress_start_year(congress)}-01-01T00:00:00Z"
    start = row[0] - WATERMARK_OVERLAP
    return start.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_run(conn: psycopg.Connection, congress: int, fetched_at: datetime) -> None:
    conn.execute(
        """
        INSERT INTO ingest_runs (congress, last_fetched_at) VALUES (%s, %s)
        ON CONFLICT (congress) DO UPDATE SET last_fetched_at = EXCLUDED.last_fetched_at
        """,
        (congress, fetched_at),
    )


def upsert_bills(conn: psycopg.Connection, records: list[BillRecord]) -> None:
    if not records:
        return
    vectors = embeddings.embed_texts([embedding_text(r) for r in records])
    with conn.cursor() as cur:
        # COALESCE keeps stored values when an incremental record lacks them
        # (a changed bill fetched without its unchanged summary, or vice versa).
        cur.executemany(
            """
            INSERT INTO bills
                (bill_id, bill_type, bill_number, congress, title, summary,
                 latest_action, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (bill_id) DO UPDATE SET
                title = EXCLUDED.title,
                summary = COALESCE(EXCLUDED.summary, bills.summary),
                latest_action = COALESCE(EXCLUDED.latest_action, bills.latest_action),
                embedding = EXCLUDED.embedding
            """,
            [
                (
                    record.bill_id,
                    record.bill_type,
                    record.number,
                    record.congress,
                    record.title,
                    record.summary,
                    record.latest_action,
                    vector,
                )
                for record, vector in zip(records, vectors, strict=True)
            ],
        )


def ingest(
    conn: psycopg.Connection,
    client: CongressGovClient,
    congress: int,
    max_bills: int | None = None,
    refresh: bool = False,
) -> int:
    run_started = datetime.now(UTC)
    since = fetch_window_start(conn, congress, refresh)
    print(f"Fetching bills for the {congress}th Congress changed since {since}...")
    bills = fetch_bills(client, congress, since, max_items=max_bills)
    print(f"  {len(bills)} bills fetched.")

    print("Fetching CRS summaries...")
    merge_summaries(bills, fetch_summaries(client, congress, since, max_items=max_bills))
    with_summary = sum(1 for b in bills.values() if b.summary)
    print(f"  {len(bills)} bills to store ({with_summary} carrying summaries).")

    records = list(bills.values())
    for start in range(0, len(records), EMBED_BATCH):
        batch = records[start : start + EMBED_BATCH]
        upsert_bills(conn, batch)
        print(f"  embedded and stored {min(start + EMBED_BATCH, len(records))}/{len(records)}")

    # Only advance the high-water mark on complete (un-limited) runs.
    if max_bills is None:
        record_run(conn, congress, run_started)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest congress.gov bills into WatchBot.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max bills and summaries to fetch per endpoint (smoke testing; "
        "does not advance the incremental high-water mark)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore the incremental high-water mark: re-fetch and re-embed the whole Congress",
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
