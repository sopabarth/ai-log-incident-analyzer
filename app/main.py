"""
FastAPI app - parse -> dedup check -> (LLM call | reuse cached analysis) -> response.

Dedup runs before the LLM call: if the same normalized error was seen
within the dedup window, we skip the model entirely and just bump the
existing incident's occurrence counter. No batch endpoint, no task
decomposition beyond parsing yet - those are added in later steps.

Schema is created/updated via Alembic migrations (`alembic upgrade head`),
not at app startup - see alembic/.
"""

from fastapi import Depends, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.dedup import find_active_incident, record_duplicate, create_new_incident
from app.llm_client import LLMResponseError, analyze_incident
from app.parser import compute_error_hash, normalize_raw_text
from app.schemas import IncidentAnalysis, IncidentRecord, RawLogEntry

app = FastAPI(title="AI Log Incident Analyzer")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/analyze-incident", response_model=IncidentRecord)
async def analyze_incident_endpoint(
    entry: RawLogEntry, session: AsyncSession = Depends(get_session)
) -> IncidentRecord:
    normalized_text = normalize_raw_text(entry.raw_text)
    if not normalized_text:
        raise HTTPException(status_code=400, detail="raw_text has no usable content")

    error_hash = compute_error_hash(normalized_text)

    existing = await find_active_incident(session, error_hash)
    if existing is not None:
        existing = await record_duplicate(session, existing)
        incident_record = IncidentRecord(
            id=existing.id,
            service=existing.service,
            environment=existing.environment,
            timestamp=existing.timestamp,
            raw_text_hash=existing.raw_text_hash,
            analysis=IncidentAnalysis(
                category=existing.category,
                root_cause_summary=existing.root_cause_summary,
                priority=existing.priority,
                priority_reasoning=existing.priority_reasoning,
                confidence=existing.confidence,
                needs_human_review=existing.needs_human_review,
            ),
            is_duplicate=True,
            duplicate_of_id=existing.id,
            occurrence_count=existing.occurrence_count,
            llm_retry_count=existing.llm_retry_count,
            llm_latency_ms=existing.llm_latency_ms,
        )

        return incident_record

    try:
        analysis, latency_ms, retry_count = await run_in_threadpool(
            analyze_incident, entry.service, entry.environment, normalized_text
        )
    except LLMResponseError as e:
        # Kept as a safety net, though analyze_incident now absorbs malformed
        # output into a fallback analysis internally - this only fires for
        # something unexpected escaping that retry loop.
        raise HTTPException(status_code=502, detail=f"LLM returned an unusable response: {e}")

    incident = await create_new_incident(
        session,
        service=entry.service,
        environment=entry.environment,
        timestamp=entry.timestamp,
        raw_text_hash=error_hash,
        category=analysis.category,
        root_cause_summary=analysis.root_cause_summary,
        priority=analysis.priority,
        priority_reasoning=analysis.priority_reasoning,
        confidence=analysis.confidence,
        needs_human_review=analysis.needs_human_review,
        llm_latency_ms=latency_ms,
        llm_retry_count=retry_count,
    )

    return IncidentRecord(
        id=incident.id,
        service=incident.service,
        environment=incident.environment,
        timestamp=incident.timestamp,
        raw_text_hash=incident.raw_text_hash,
        analysis=analysis,
        is_duplicate=False,
        duplicate_of_id=None,
        occurrence_count=incident.occurrence_count,
        llm_retry_count=incident.llm_retry_count,
        llm_latency_ms=incident.llm_latency_ms,
    )
