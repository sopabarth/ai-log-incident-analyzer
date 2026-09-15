"""
Async Postgres layer - engine, session, and the Incident table.

Schema is managed by Alembic (see alembic/) rather than an ad-hoc
create_all() call - run `alembic upgrade head` to create/update tables.
"""

import os
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from sqlalchemy import Boolean, DateTime, Enum, Float, Index, Integer, String, Uuid
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.schemas import ErrorCategory, Environment, Priority


def _enum_values(enum_cls):
    """
    By default SQLAlchemy persists a Python Enum member's .name
    ("CRITICAL"), not its .value ("critical"). Every enum column below
    passes this as values_callable so the stored strings match what the
    rest of the app (JSON API, Pydantic) already uses.
    """
    return [member.value for member in enum_cls]

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

    id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    service: Mapped[str] = mapped_column(String, nullable=False)
    environment: Mapped[Environment] = mapped_column(
        Enum(Environment, values_callable=_enum_values, name="environment_enum"),
        nullable=False,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    # category is VARCHAR in db with CHECK constraint and not a native enum
    category: Mapped[ErrorCategory] = mapped_column(
        Enum(
            ErrorCategory,
            native_enum=False,
            create_constraint=True,
            values_callable=_enum_values,
            name="ck_incidents_category",
        ),
        nullable=False,
    )
    root_cause_summary: Mapped[str] = mapped_column(String(300), nullable=False)
    priority: Mapped[Priority] = mapped_column(
        Enum(Priority, values_callable=_enum_values, name="priority_enum"),
        nullable=False,
    )
    priority_reasoning: Mapped[str] = mapped_column(String(200), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    needs_human_review: Mapped[bool] = mapped_column(Boolean, nullable=False)

    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    llm_retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


async def get_session() -> AsyncGenerator[AsyncSession, Any]:
    """FastAPI dependency - yields one session per request."""
    async with SessionLocal() as session:
        yield session
