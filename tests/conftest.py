"""Shared fixtures.

DB tests run against a dedicated ``<dbname>_test`` database so they never
touch the application's bills index. It is dropped and recreated once per
session, so schema changes always apply (``CREATE TABLE IF NOT EXISTS``
would silently skip them on a stale database).
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
    try:
        admin.execute(f'DROP DATABASE IF EXISTS "{test_name}" WITH (FORCE)')
        admin.execute(f'CREATE DATABASE "{test_name}"')
    except psycopg.Error as err:
        pytest.skip(f"Cannot recreate test database {test_name!r}: {err}")
    finally:
        admin.close()
    return f"{base}/{test_name}"
