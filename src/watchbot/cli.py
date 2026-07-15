"""WatchBot chat REPL."""

from __future__ import annotations

import asyncio
import contextlib
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich.console import Console

from watchbot import db
from watchbot.agent import WatchBotAgent, build_tools
from watchbot.config import settings

console = Console()

BANNER = """\
[bold green]WatchBot[/bold green] -- your watchdog on the {congress}th Congress.
Ask about bills by topic, how members voted, or voting-record comparisons.
Type [bold]exit[/bold] to quit.
"""


async def chat() -> None:
    config = settings()

    try:
        conn = db.connect(config.database_url)
    except Exception as err:
        sys.exit(
            f"Could not connect to Postgres ({err}).\n"
            "Is the database up? Try: docker compose up -d"
        )

    bill_count = conn.execute(
        "SELECT count(*) FROM bills WHERE congress = %s", (config.congress,)
    ).fetchone()[0]
    if bill_count == 0:
        console.print(
            "[yellow]The bills index is empty -- run `uv run watchbot-ingest` first.\n"
            "Continuing anyway (vote and legislator questions will still work).[/yellow]"
        )

    argv = config.govql_mcp_argv
    server = StdioServerParameters(command=argv[0], args=argv[1:])

    async with stdio_client(server) as (read, write), ClientSession(read, write) as mcp_session:
        await mcp_session.initialize()
        tools = await build_tools(conn, mcp_session)
        agent = WatchBotAgent(tools, congress=config.congress)

        console.print(BANNER.format(congress=config.congress))
        console.print(f"[dim]Connected to GovQL MCP ({len(tools) - 1} tools) + local bill index "
                      f"({bill_count} bills).[/dim]\n")

        while True:
            try:
                user_input = (await asyncio.to_thread(console.input, "[bold]You:[/bold] ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                break

            console.print("\n[bold blue]WatchBot:[/bold blue] ", end="")
            try:
                await agent.ask(
                    user_input,
                    on_text=lambda text: print(text, end="", flush=True),
                    on_tool_use=lambda name, args: console.print(
                        f"\n  ⚙ {name}({_short(args)})", style="dim", markup=False, highlight=False
                    ),
                )
            except Exception as err:  # keep the REPL alive on API/tool errors
                console.print(f"\n[red]Error: {err}[/red]")
            console.print("\n")

    conn.close()
    console.print("[green]Goodbye![/green]")


def _short(args: dict, max_len: int = 90) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def main() -> None:
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(chat())


if __name__ == "__main__":
    main()
