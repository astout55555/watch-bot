# WatchBot

A command-line chatbot that answers questions about the current (119th) US Congress with
grounded data instead of guesses. Ask it about legislation by topic, how specific members
voted, or which members break with their party, and it cites the bills and roll-call votes
it used.

```
You: How did the House vote this week on the purchase-privacy bill?

  ⚙ search_bills(query='purchase privacy')
  ⚙ execute_graphql(query='{ allVotes(filter: {question: {includesInsensitive: "H R 1181"}}) ... }')

WatchBot: The House passed H.R. 1181 (hr1181-119), the Protecting Privacy in
Purchases Act, on July 14, 2026 (roll call h240-119.2026). ...
```

## How it works

Language models are unreliable about Congress: they fabricate bill numbers and can't know
about votes cast after their training data ends. WatchBot avoids both problems by giving
Claude two retrieval tools and instructing it never to answer from memory:

1. **A local semantic index of bills.** An ingest script pulls every 119th-Congress bill
   (title + CRS summary) from the [congress.gov API](https://api.congress.gov/) and embeds
   it into Postgres/pgvector. This answers the discovery question that keyword search
   handles badly: "which bills are actually about X?"
2. **The [GovQL](https://govql.us) MCP server.** GovQL exposes structured congressional
   data (roll-call votes, individual member positions, party-agreement analytics) as a
   GraphQL API, refreshed daily. The agent connects to its MCP server and queries it live,
   so every tally comes from the official record and answers stay current.

The two halves are joined by GovQL's canonical bill id (`hr1181-119`). GovQL's own bills
table isn't populated yet, so today the agent finds a bill's votes by searching vote
question text for the bill's reference forms ("H R 1181" / "H.R. 1181"); once GovQL ships
bill data, the same ids join directly via `relatedBillId`.

## Setup

You'll need Docker, [uv](https://docs.astral.sh/uv/), and three API keys: Anthropic
(the chat agent runs on Claude), OpenAI (embeddings only), and congress.gov
([free signup](https://api.congress.gov/sign-up/)).

```bash
git clone <this repo> && cd watch-bot
cp .env.example .env        # then fill in your keys
docker compose up -d        # Postgres + pgvector on port 5433
uv sync
uv run watchbot-setup-db
uv run watchbot-ingest      # full Congress; try --limit 200 for a quick first pass
uv run watchbot
```

Re-running `watchbot-ingest` only embeds bills it hasn't seen; pass `--refresh` to
re-embed everything (for example after a batch of new CRS summaries lands).

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_API_KEY` | Chat agent | required |
| `OPENAI_API_KEY` | Embeddings (`text-embedding-3-small`) | required |
| `CONGRESS_GOV_API_KEY` | Bill ingest | required |
| `WATCHBOT_MODEL` | Which Claude model runs the agent | `claude-haiku-4-5` |
| `DATABASE_URL` | Postgres + pgvector | the compose.yml container |
| `GOVQL_MCP_COMMAND` | How to launch the GovQL MCP server (any stdio MCP command works) | `uvx govql-mcp-server` |
| `CONGRESS` | Which Congress to ingest and discuss | `119` |

## Development

```bash
uv run pytest        # DB integration tests skip themselves if the container is down
uv run ruff check .
```

Tests stub all embedding calls and never touch the Anthropic or OpenAI APIs.

## Project layout

| Module | Role |
|---|---|
| `billref.py` | Converts between bill id forms: congress.gov types, GovQL canonical ids, vote-question text |
| `ingest.py` | congress.gov bulk fetch, summary merging, embedding, upsert |
| `db.py` / `search.py` | Schema and cosine search over the bill index |
| `agent.py` | Claude tool runner: local `search_bills` tool + GovQL MCP tools |
| `cli.py` | Chat REPL |
