import pytest

from watchbot import db, embeddings
from watchbot.config import EMBEDDING_DIMENSIONS
from watchbot.ingest import (
    BillRecord,
    _retry_after_seconds,
    congress_start_year,
    embedding_text,
    merge_summaries,
    parse_bill_item,
    parse_summary_item,
    plain_text,
    upsert_bills,
)

BILL_ITEM = {
    "congress": 119,
    "latestAction": {"actionDate": "2026-07-14", "text": "Passed the House."},
    "number": "1181",
    "originChamber": "House",
    "title": "Protecting Privacy in Purchases Act",
    "type": "HR",
    "updateDate": "2026-07-15",
    "url": "https://api.congress.gov/v3/bill/119/hr/1181?format=json",
}

SUMMARY_ITEM = {
    "actionDate": "2026-05-02",
    "actionDesc": "Introduced in House",
    "bill": {
        "congress": 119,
        "number": "1181",
        "title": "Protecting Privacy in Purchases Act",
        "type": "HR",
    },
    "text": "<p>This bill <strong>prohibits</strong> certain data sales.</p>",
    "updateDate": "2026-05-03",
}


class TestParsing:
    def test_parse_bill_item(self):
        record = parse_bill_item(BILL_ITEM)
        assert record.bill_id == "hr1181-119"
        assert record.bill_type == "hr"
        assert record.number == 1181
        assert record.congress == 119
        assert record.title == "Protecting Privacy in Purchases Act"
        assert record.latest_action == "2026-07-14: Passed the House."
        assert record.summary is None

    def test_parse_summary_item_builds_standalone_record(self):
        record = parse_summary_item(SUMMARY_ITEM)
        assert record.bill_id == "hr1181-119"
        assert record.title == "Protecting Privacy in Purchases Act"
        assert record.summary == "This bill prohibits certain data sales."
        assert record._summary_date == "2026-05-02"
        assert record.latest_action is None

    def test_parse_summary_item_tolerates_null_action_date(self):
        item = {**SUMMARY_ITEM, "actionDate": None}
        assert parse_summary_item(item)._summary_date == ""

    def test_plain_text_strips_html_and_unescapes(self):
        assert plain_text("<p>A &amp; B</p><ul><li>C</li></ul>") == "A & B C"


class TestMergeSummaries:
    def _summary(self, bill_id: str, date: str, text: str) -> BillRecord:
        return BillRecord(
            bill_id=bill_id,
            bill_type="hr",
            number=1181,
            congress=119,
            title="Protecting Privacy in Purchases Act",
            summary=text,
            latest_action=None,
            _summary_date=date,
        )

    def test_latest_summary_wins(self):
        bills = {"hr1181-119": parse_bill_item(BILL_ITEM)}
        merge_summaries(
            bills,
            [
                self._summary("hr1181-119", "2026-05-02", "Old summary."),
                self._summary("hr1181-119", "2026-06-10", "Newer summary."),
            ],
        )
        assert bills["hr1181-119"].summary == "Newer summary."

    def test_summary_without_fetched_bill_becomes_standalone_record(self):
        bills: dict[str, BillRecord] = {}
        merge_summaries(bills, [self._summary("hr1181-119", "2026-06-10", "Summary.")])
        assert bills["hr1181-119"].summary == "Summary."
        assert bills["hr1181-119"].title == "Protecting Privacy in Purchases Act"


class TestEmbeddingText:
    def test_includes_display_name_title_and_summary(self):
        record = BillRecord(
            bill_id="hr1181-119",
            bill_type="hr",
            number=1181,
            congress=119,
            title="Protecting Privacy in Purchases Act",
            summary="This bill prohibits certain data sales.",
            latest_action=None,
        )
        text = embedding_text(record)
        assert "H.R. 1181" in text
        assert "Protecting Privacy in Purchases Act" in text
        assert "prohibits certain data sales" in text

    def test_truncates_very_long_summaries(self):
        record = BillRecord(
            bill_id="s1-119",
            bill_type="s",
            number=1,
            congress=119,
            title="Big Bill",
            summary="x" * 50_000,
            latest_action=None,
        )
        assert len(embedding_text(record)) <= 21_000


class TestRetryAfterSeconds:
    def test_integer_header(self):
        assert _retry_after_seconds("12") == 12

    def test_missing_header_uses_default(self):
        assert _retry_after_seconds(None) == 30

    def test_http_date_header_falls_back(self):
        assert _retry_after_seconds("Wed, 21 Oct 2026 07:28:00 GMT") == 30


class TestCongressStartYear:
    def test_known_congresses(self):
        assert congress_start_year(119) == 2025
        assert congress_start_year(118) == 2023
        assert congress_start_year(1) == 1789


class TestUpsertCoalesce:
    """Incremental runs must not null stored fields when a record lacks them."""

    @pytest.fixture()
    def conn(self, monkeypatch, test_database_url):
        connection = db.connect(test_database_url)
        db.setup(connection)
        connection.execute("TRUNCATE bills")
        monkeypatch.setattr(
            embeddings,
            "embed_texts",
            lambda texts: [[0.0] * EMBEDDING_DIMENSIONS for _ in texts],
        )
        yield connection
        connection.close()

    def test_bill_only_update_keeps_stored_summary(self, conn):
        with_summary = BillRecord(
            "hr1-119", "hr", 1, 119, "Bill One", "The summary.", "2026-01-01: Introduced."
        )
        upsert_bills(conn, [with_summary])

        bill_only = BillRecord("hr1-119", "hr", 1, 119, "Bill One (amended)", None, None)
        upsert_bills(conn, [bill_only])

        title, summary, action = conn.execute(
            "SELECT title, summary, latest_action FROM bills WHERE bill_id = 'hr1-119'"
        ).fetchone()
        assert title == "Bill One (amended)"
        assert summary == "The summary."
        assert action == "2026-01-01: Introduced."
