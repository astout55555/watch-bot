"""Environment-backed settings shared across WatchBot modules."""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

def _env(name: str, default: str) -> str:
    """Read an env var, treating empty values (e.g. a blank .env line) as unset."""
    return os.environ.get(name) or default


EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
CHAT_MODEL = _env("WATCHBOT_MODEL", "claude-haiku-4-5")


@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: _env(
            "DATABASE_URL", "postgresql://postgres:watchbot@localhost:5433/watchbot"
        )
    )
    congress: int = field(default_factory=lambda: int(_env("CONGRESS", "119")))
    congress_gov_api_key: str = field(default_factory=lambda: _env("CONGRESS_GOV_API_KEY", ""))
    govql_mcp_command: str = field(
        default_factory=lambda: _env("GOVQL_MCP_COMMAND", "uvx govql-mcp-server")
    )

    @property
    def govql_mcp_argv(self) -> list[str]:
        return shlex.split(self.govql_mcp_command)


def settings() -> Settings:
    return Settings()
