"""
Dedup / idempotency layer.

Goal: if the same normalized error shows up hundreds of times in a burst
(a real scenario - one root cause, many requests hitting it), don't call
the LLM for every single occurrence. Instead, look up the error's hash in
Postgres; if a matching incident was last seen within DEDUP_WINDOW_MINUTES,
treat this as a duplicate (bump its counters) and skip the LLM entirely.
If the window has expired, treat it as a fresh incident - the same bug
resurfacing after a gap plausibly deserves a new analysis, not a stale one.

This intentionally lives against the database rather than an in-process
cache: an in-memory dict wouldn't survive a restart and wouldn't be shared
across multiple worker processes, both of which would silently break dedup
correctness (and defeat the point of avoiding repeat LLM calls).
"""

import os
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import Incident

DEFAULT_DEDUP_WINDOW_MINUTES = 10


def _window_minutes() -> int:
    return int(os.getenv("DEDUP_WINDOW_MINUTES", DEFAULT_DEDUP_WINDOW_MINUTES))


async def find_active_incident(session: AsyncSession, raw_text_hash: str) -> Incident | None:
    """
    Return the most recently seen incident with this hash, if it was last
    seen within the dedup window - otherwise None (treat as a new incident).
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=_window_minutes())
    stmt = (
        select(Incident)
        .where(Incident.raw_text_hash == raw_text_hash, Incident.last_seen_at >= cutoff)
        .order_by(Incident.last_seen_at.desc())
        .limit(1)
    )
    result = await session.execute(stmt)
    return result.scalar_one_or_none()


async def record_duplicate(session: AsyncSession, incident: Incident) -> Incident:
    """Bump occurrence_count/last_seen_at on an existing incident - no LLM call."""
    incident.occurrence_count += 1
    incident.last_seen_at = datetime.now(timezone.utc)
    await session.commit()
    await session.refresh(incident)
    return incident


async def create_new_incident(
    session: AsyncSession,
    *,
    service: str,
    environment: str,
    timestamp: datetime,
    raw_text_hash: str,
    category: str,
    root_cause_summary: str,
    priority: str,
    priority_reasoning: str,
    confidence: float,
    needs_human_review: bool,
    llm_latency_ms: int | None,
) -> Incident:
    """Insert a fresh incident row after a real LLM call."""
    now = datetime.now(timezone.utc)
    incident = Incident(
        service=service,
        environment=environment,
        timestamp=timestamp,
        raw_text_hash=raw_text_hash,
        category=category,
        root_cause_summary=root_cause_summary,
        priority=priority,
        priority_reasoning=priority_reasoning,
        confidence=confidence,
        needs_human_review=needs_human_review,
        occurrence_count=1,
        llm_retry_count=0,
        llm_latency_ms=llm_latency_ms,
        first_seen_at=now,
        last_seen_at=now,
    )
    session.add(incident)
    await session.commit()
    await session.refresh(incident)
    return incident
