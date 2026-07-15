"""Environment-backed settings shared across WatchBot modules."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
CHAT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql://postgres:watchbot@localhost:5433/watchbot"
        )
    )
    congress: int = field(default_factory=lambda: int(os.environ.get("CONGRESS", "119")))
    congress_gov_api_key: str = field(
        default_factory=lambda: os.environ.get("CONGRESS_GOV_API_KEY", "")
    )
    govql_mcp_command: str = field(
        default_factory=lambda: os.environ.get("GOVQL_MCP_COMMAND", "uvx govql-mcp-server")
    )

    @property
    def govql_mcp_argv(self) -> list[str]:
        return shlex.split(self.govql_mcp_command)


def settings() -> Settings:
    return Settings()
