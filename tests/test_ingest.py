from watchbot.ingest import (
    BillRecord,
    embedding_text,
    merge_summaries,
    parse_bill_item,
    parse_summary_item,
    plain_text,
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

    def test_parse_summary_item(self):
        bill_id, action_date, text = parse_summary_item(SUMMARY_ITEM)
        assert bill_id == "hr1181-119"
        assert action_date == "2026-05-02"
        assert text == "This bill prohibits certain data sales."

    def test_plain_text_strips_html_and_unescapes(self):
        assert plain_text("<p>A &amp; B</p><ul><li>C</li></ul>") == "A & B C"


class TestMergeSummaries:
    def test_latest_summary_wins(self):
        bills = {"hr1181-119": parse_bill_item(BILL_ITEM)}
        summaries = [
            ("hr1181-119", "2026-05-02", "Old summary."),
            ("hr1181-119", "2026-06-10", "Newer summary."),
            ("s999-119", "2026-06-01", "No matching bill; ignored."),
        ]
        merge_summaries(bills, summaries)
        assert bills["hr1181-119"].summary == "Newer summary."


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
