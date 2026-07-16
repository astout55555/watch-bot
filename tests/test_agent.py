"""Tests for the agent's local tool. No Anthropic/OpenAI calls are made."""

import json

import pytest

from watchbot import db, embeddings
from watchbot.agent import SYSTEM_PROMPT, make_search_bills_tool
from watchbot.config import EMBEDDING_DIMENSIONS


@pytest.fixture()
def conn(monkeypatch, test_database_url):
    connection = db.connect(test_database_url)
    db.setup(connection)
    connection.execute("TRUNCATE bills")
    embedding = [0.0] * EMBEDDING_DIMENSIONS
    embedding[0] = 1.0
    connection.execute(
        """
        INSERT INTO bills (bill_id, bill_type, bill_number, congress, title, summary,
                           latest_action, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            "hr1181-119",
            "hr",
            1181,
            119,
            "Protecting Privacy in Purchases Act",
            "This bill prohibits certain data sales.",
            "2026-07-14: Passed the House.",
            embedding,
        ),
    )
    monkeypatch.setattr(embeddings, "embed_query", lambda text: embedding)
    yield connection
    connection.close()


class TestSearchBillsTool:
    async def test_returns_json_with_vote_question_refs(self, conn):
        tool = make_search_bills_tool(conn, congress=119)
        result = json.loads(await tool.call({"query": "purchase privacy"}))
        (hit,) = result["results"]
        assert hit["bill_id"] == "hr1181-119"
        assert hit["congress"] == 119
        assert hit["title"] == "Protecting Privacy in Purchases Act"
        assert "H R 1181" in hit["vote_question_refs"]
        assert "H.R. 1181" in hit["vote_question_refs"]
        assert hit["latest_action"] == "2026-07-14: Passed the House."

    def test_tool_schema_generated_from_signature(self, conn):
        tool = make_search_bills_tool(conn, congress=119)
        definition = tool.to_dict()
        assert definition["name"] == "search_bills"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]


class TestSystemPrompt:
    def test_formats_with_congress_number(self):
        prompt = SYSTEM_PROMPT.format(congress=119)
        assert "119th Congress" in prompt
        assert "{" not in prompt.replace("{congress}", "")
