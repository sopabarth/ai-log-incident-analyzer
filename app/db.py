"""
Async Postgres layer - engine, session, and the Incident table.

No Alembic/migrations yet (bare-bones on purpose, same as the rest of the
project so far): tables are created directly from the ORM metadata on
startup via init_db(). Swap in real migrations once the schema needs to
evolve without dropping data.
"""

import os
from datetime import datetime
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://root:root@localhost:5432/incident_analyzer",
)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class Incident(Base):
    """
    One row per *distinct* incident (not one row per occurrence).

    When the same normalized error re-arrives within the dedup window
    (see app/dedup.py), we update occurrence_count/last_seen_at on the
    existing row instead of inserting a new one and re-calling the LLM.
    raw_text_hash is indexed (not unique) since the same hash legitimately
    gets a new row once the dedup window has expired.
    """

    __tablename__ = "incidents"
    __table_args__ = (Index("ix_incidents_raw_text_hash", "raw_text_hash"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    service: Mapped[str] = mapped_column(String, nullable=False)
    environment: Mapped[str] = mapped_column(String, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    category: Mapped[str] = mapped_column(String, nullable=False)
    root_cause_summary: Mapped[str] = mapped_column(String(300), nullable=False)
    priority: Mapped[str] = mapped_column(String, nullable=False)
    priority_reasoning: Mapped[str] = mapped_column(String(200), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, nullable=False)

    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    llm_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


async def init_db() -> None:
    """Create tables if they don't exist yet. Called once on app startup."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_session() -> AsyncGenerator[AsyncSession, Any]:
    """FastAPI dependency - yields one session per request."""
    async with SessionLocal() as session:
        yield session
