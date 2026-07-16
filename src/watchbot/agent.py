"""The WatchBot agent: a Claude tool loop over two retrieval surfaces.

1. ``search_bills`` -- semantic search over our local index of current-Congress
   bills (titles + CRS summaries in pgvector).
2. The GovQL MCP server's tools -- structured roll-call votes, legislators,
   and voting analytics, converted for the tool runner via ``async_mcp_tool``.

The bridge between the two is textual for now: GovQL's ``bills`` table is
empty, so the agent finds a bill's roll-call votes by searching vote
``question`` text for the bill's reference forms (see ``billref``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import psycopg
from anthropic import AsyncAnthropic
from anthropic.lib.tools import beta_async_tool
from anthropic.lib.tools.mcp import async_mcp_tool
from mcp import ClientSession

from watchbot import search
from watchbot.billref import parse_bill_id, question_reference_variants
from watchbot.config import CHAT_MODEL

MAX_TOKENS = 16_000
MAX_TOOL_ITERATIONS = 15

SYSTEM_PROMPT = """\
You are WatchBot, a nonpartisan congressional watchdog assistant covering the \
{congress}th Congress. You help people find out what Congress is actually doing: \
which bills exist on a topic, how members voted, and how members' voting records compare.

You have two kinds of tools:

1. `search_bills` -- semantic search over an index of {congress}th-Congress bills \
(titles and CRS summaries). Use it whenever the user asks about legislation by TOPIC \
rather than by number. Each result includes the bill's canonical id (e.g. hr1181-119) \
and `vote_question_refs`: the exact text forms that identify the bill inside GovQL \
vote questions.
2. GovQL tools -- live structured data on roll-call votes, legislators, and voting \
analytics for the US Congress.

Linking bills to votes: GovQL's bills table is not yet populated, so `relatedBillId` \
on votes is null. To find roll-call votes on a bill, search vote question text for the \
bill's `vote_question_refs`. Vote questions write bill references in TWO styles -- \
House clerk style with spaces ("H R 1181") and dotted style ("H.R. 1181") -- and a \
substring search on one will NOT match the other, so try the spaced form first and \
fall back to the dotted form before concluding a bill has no votes. If a topic-search \
tool over votes is available, use a ref as the topic; otherwise filter votes with \
GraphQL using a case-insensitive "question contains" condition. If `relatedBillId` \
ever comes back non-null, prefer it.

Grounding rules:
- Never state a bill number, vote tally, or member position from memory -- only from \
tool results. If tools don't return it, say you couldn't find it.
- Cite what you use: bills by their display form and id (H.R. 1181 / hr1181-119), \
votes by their vote id and, when available, their official source URL.
- Stay neutral: report voting records and outcomes without editorializing.
- Keep answers conversational and tight -- a paragraph or two, or a short list when \
comparing members or votes. Lead with the answer, not your process.
"""


def make_search_bills_tool(conn: psycopg.Connection, congress: int):
    """Build the local semantic-search tool bound to a database connection."""

    @beta_async_tool
    async def search_bills(query: str, limit: int = 8) -> str:
        """Semantically search bills of the current Congress by topic.

        Returns the closest-matching bills with their canonical bill id, title,
        CRS summary excerpt, latest action, and the `vote_question_refs` text
        forms used to locate the bill's roll-call votes in GovQL.

        Args:
            query: A natural-language topic, e.g. "surprise medical billing".
            limit: Maximum number of bills to return (default 8).
        """
        hits = await asyncio.to_thread(search.search_bills, conn, query, congress, limit)
        results = []
        for hit in hits:
            ref = parse_bill_id(hit.bill_id)
            summary = hit.summary or ""
            results.append(
                {
                    "bill_id": hit.bill_id,
                    "congress": hit.congress,
                    "title": hit.title,
                    "summary": summary[:1200] + ("..." if len(summary) > 1200 else ""),
                    "latest_action": hit.latest_action,
                    "similarity": round(hit.similarity, 3),
                    "vote_question_refs": question_reference_variants(ref),
                }
            )
        return json.dumps({"results": results})

    return search_bills


async def build_tools(
    conn: psycopg.Connection, mcp_session: ClientSession, congress: int
) -> list:
    tools_result = await mcp_session.list_tools()
    return [make_search_bills_tool(conn, congress)] + [
        async_mcp_tool(tool, mcp_session) for tool in tools_result.tools
    ]


class WatchBotAgent:
    """Holds conversation history and runs one agentic turn per user message."""

    def __init__(
        self,
        tools: list,
        congress: int,
        client: AsyncAnthropic | None = None,
        model: str = CHAT_MODEL,
    ):
        self._client = client or AsyncAnthropic()
        self._tools = tools
        self._model = model
        self._system = SYSTEM_PROMPT.format(congress=congress)
        self._messages: list = []

    async def ask(
        self,
        user_input: str,
        on_text: Callable[[str], None],
        on_tool_use: Callable[[str, dict], None],
    ) -> None:
        """Run one turn: stream text via `on_text`, report tool calls via `on_tool_use`."""
        self._messages.append({"role": "user", "content": user_input})

        runner = self._client.beta.messages.tool_runner(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=self._system,
            tools=self._tools,
            messages=self._messages,
            max_iterations=MAX_TOOL_ITERATIONS,
            stream=True,
        )

        async for stream in runner:
            async with stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        on_text(event.delta.text)
                    elif (
                        event.type == "content_block_stop"
                        and event.content_block.type == "tool_use"
                    ):
                        on_tool_use(event.content_block.name, event.content_block.input)
                message = await stream.get_final_message()

            # The runner keeps its own history; mirror it so the next user
            # turn continues the same conversation.
            self._messages.append({"role": "assistant", "content": message.content})
            tool_response = await runner.generate_tool_call_response()
            if tool_response is not None:
                self._messages.append(tool_response)
