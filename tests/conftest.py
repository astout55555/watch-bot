"""Shared fixtures.

DB tests run against a dedicated ``<dbname>_test`` database (created on demand)
so they never touch the application's bills index.
"""

import psycopg
import pytest

from watchbot.config import settings


@pytest.fixture(scope="session")
def test_database_url() -> str:
    url = settings().database_url
    base, _, name = url.rpartition("/")
    test_name = f"{name}_test"
    try:
        admin = psycopg.connect(url, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres (compose.yml container) is not running")
    exists = admin.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s", (test_name,)
    ).fetchone()
    if not exists:
        admin.execute(f'CREATE DATABASE "{test_name}"')
    admin.close()
    return f"{base}/{test_name}"
