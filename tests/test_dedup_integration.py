"""
Dedup and the schema behind it, against a real Postgres.

Skipped unless TEST_DATABASE_URL is set, e.g.

    TEST_DATABASE_URL=postgresql+asyncpg://root:root@localhost:5432/incident_analyzer uv run pytest

The URL only says which server to use: the tests create their own database
next to the one it names, build the schema in it by running the real Alembic
migrations, and drop it at the end - the development data is never touched.
"""

import asyncio
import os
import uuid
from datetime import datetime, timedelta, timezone

import asyncpg
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

import app.db
from app.dedup import create_new_incident, find_active_incident, record_duplicate
from app.schemas import Environment, ErrorCategory, Priority

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL is not set"),
]


def asyncpg_dsn(url) -> str:
    """The SQLAlchemy URL as a plain postgresql:// DSN for asyncpg."""
    return url.set(drivername="postgresql").render_as_string(hide_password=False)


def alembic_config() -> Config:
    return Config("alembic.ini")


@pytest.fixture(scope="session")
def database_url():
    """A fresh database with the schema built by the real migrations."""
    server = make_url(TEST_DATABASE_URL)
    name = f"incident_analyzer_test_{uuid.uuid4().hex[:8]}"
    test_url = server.set(database=name)

    async def admin(sql: str):
        conn = await asyncpg.connect(asyncpg_dsn(server))
        try:
            await conn.execute(sql)
        finally:
            await conn.close()

    asyncio.run(admin(f'CREATE DATABASE "{name}"'))
    url = test_url.render_as_string(hide_password=False)
    # alembic/env.py takes its URL from app.db.DATABASE_URL.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(app.db, "DATABASE_URL", url)
        command.upgrade(alembic_config(), "head")
        yield url
    asyncio.run(admin(f'DROP DATABASE "{name}" WITH (FORCE)'))


@pytest.fixture
async def sessions(database_url):
    """A session factory on an empty incidents table."""
    # NullPool: no connections outlive the test's event loop.
    engine = create_async_engine(database_url, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE incidents"))
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def new_incident(session: AsyncSession, raw_text_hash: str = "a" * 64, **overrides):
    fields = dict(service="checkout-api", environment=Environment.PROD,
                  timestamp=datetime(2026, 11, 24, 19, tzinfo=timezone.utc), raw_text_hash=raw_text_hash,
                  category=ErrorCategory.DATABASE_TIMEOUT, root_cause_summary="Pool exhausted.",
                  priority=Priority.CRITICAL, priority_reasoning="All checkout requests fail.", confidence=0.95,
                  needs_human_review=False, llm_latency_ms=1234, llm_retry_count=1)
    return await create_new_incident(session, **{**fields, **overrides})


async def last_seen_minutes_ago(session: AsyncSession, incident_id, minutes: int, column: str = "last_seen_at"):
    await session.execute(text(f"UPDATE incidents SET {column} = now() - make_interval(mins => :m) WHERE id = :id"),
                          {"m": minutes, "id": incident_id})
    await session.commit()


# --- create / find ---

@pytest.mark.anyio
async def test_create_new_incident_stores_a_fresh_row(sessions):
    async with sessions() as s:
        incident = await new_incident(s)

    async with sessions() as s:
        stored = await find_active_incident(s, "a" * 64)
    assert stored.id == incident.id
    assert stored.environment is Environment.PROD and stored.priority is Priority.CRITICAL
    assert stored.category is ErrorCategory.DATABASE_TIMEOUT
    assert stored.occurrence_count == 1
    assert (stored.llm_retry_count, stored.llm_latency_ms) == (1, 1234)
    assert stored.first_seen_at == stored.last_seen_at


@pytest.mark.anyio
async def test_nothing_is_found_for_an_unknown_hash(sessions):
    async with sessions() as s:
        await new_incident(s, raw_text_hash="a" * 64)
        assert await find_active_incident(s, "b" * 64) is None


@pytest.mark.anyio
async def test_window_boundary(sessions, monkeypatch):
    monkeypatch.setenv("DEDUP_WINDOW_MINUTES", "10")
    async with sessions() as s:
        incident = await new_incident(s)
        await last_seen_minutes_ago(s, incident.id, 9)
        assert await find_active_incident(s, "a" * 64) is not None
        await last_seen_minutes_ago(s, incident.id, 11)
        assert await find_active_incident(s, "a" * 64) is None


@pytest.mark.anyio
async def test_window_length_comes_from_the_environment(sessions, monkeypatch):
    async with sessions() as s:
        incident = await new_incident(s)
        await last_seen_minutes_ago(s, incident.id, 30)
        monkeypatch.setenv("DEDUP_WINDOW_MINUTES", "10")
        assert await find_active_incident(s, "a" * 64) is None
        monkeypatch.setenv("DEDUP_WINDOW_MINUTES", "60")
        assert await find_active_incident(s, "a" * 64) is not None


@pytest.mark.anyio
async def test_the_most_recently_seen_incident_wins(sessions):
    # The same error resurfacing after the window gets a new row, so one hash
    # can have several; the current one is the most recently seen.
    async with sessions() as s:
        older = await new_incident(s)
        newer = await new_incident(s)
        await last_seen_minutes_ago(s, older.id, 2)
        await last_seen_minutes_ago(s, newer.id, 1)
        assert (await find_active_incident(s, "a" * 64)).id == newer.id


# --- duplicates ---

@pytest.mark.anyio
async def test_record_duplicate_counts_and_slides_the_window(sessions, monkeypatch):
    monkeypatch.setenv("DEDUP_WINDOW_MINUTES", "10")
    async with sessions() as s:
        incident = await new_incident(s)
        await last_seen_minutes_ago(s, incident.id, 30, column="first_seen_at")
        await last_seen_minutes_ago(s, incident.id, 5)

        found = await find_active_incident(s, "a" * 64)
        updated = await record_duplicate(s, found)

    assert updated.occurrence_count == 2
    assert datetime.now(timezone.utc) - updated.last_seen_at < timedelta(seconds=5)
    # The window is measured from the last sighting, not the first: a steady
    # stream of the same error keeps hitting the same incident.
    assert datetime.now(timezone.utc) - updated.first_seen_at > timedelta(minutes=29)


@pytest.mark.anyio
async def test_concurrent_duplicates_are_all_counted(sessions):
    """A burst of the same error is exactly the case dedup exists for, so
    simultaneous duplicates must each be counted, none lost."""
    async with sessions() as s:
        incident = await new_incident(s)

    async def one_duplicate():
        async with sessions() as s:
            found = await find_active_incident(s, "a" * 64)
            await record_duplicate(s, found)

    await asyncio.gather(*(one_duplicate() for _ in range(20)))

    async with sessions() as s:
        stored = await find_active_incident(s, "a" * 64)
    assert stored.id == incident.id
    assert stored.occurrence_count == 21


# --- schema built by the migrations ---

@pytest.mark.anyio
@pytest.mark.parametrize("column, bad_value", [
    ("category", "made_up_category"),  # CHECK constraint
    ("environment", "staging_v2"),     # native enum
    ("priority", "urgent"),            # native enum
])
async def test_database_rejects_values_outside_the_enums(sessions, column, bad_value):
    """The constraints hold even for writes that bypass the app's validation,
    like a manual UPDATE."""
    async with sessions() as s:
        incident = await new_incident(s)
        with pytest.raises(DBAPIError):
            await s.execute(text(f"UPDATE incidents SET {column} = :v WHERE id = :id"),
                            {"v": bad_value, "id": incident.id})


def test_migrations_downgrade_to_nothing_and_back(database_url, monkeypatch):
    async def tables() -> set[str]:
        conn = await asyncpg.connect(asyncpg_dsn(make_url(database_url)))
        try:
            rows = await conn.fetch("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
            # Enum types (typtype 'e') in the public schema only, not the system catalog's.
            types = await conn.fetch("SELECT t.typname FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace "
                                     "WHERE n.nspname = 'public' AND t.typtype = 'e'")
            return {r["tablename"] for r in rows} | {r["typname"] for r in types}
        finally:
            await conn.close()

    monkeypatch.setattr(app.db, "DATABASE_URL", database_url)
    assert {"incidents", "environment_enum", "priority_enum"} <= asyncio.run(tables())

    command.downgrade(alembic_config(), "base")
    assert asyncio.run(tables()) == {"alembic_version"}

    command.upgrade(alembic_config(), "head")
    assert {"incidents", "environment_enum", "priority_enum"} <= asyncio.run(tables())
