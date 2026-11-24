"""
FastAPI app - parse -> dedup check -> (LLM call | reuse cached analysis) -> response.

Dedup runs before the LLM call: if the same normalized error was seen
within the dedup window, we skip the model entirely and just bump the
existing incident's occurrence counter. No batch endpoint, no task
decomposition beyond parsing yet - those are added in later steps.

Schema is created/updated via Alembic migrations (`alembic upgrade head`),
not at app startup - see alembic/.

Each fresh analysis also leaves a trace of its model calls in MongoDB (see
app/mongo.py). It's written as a background task, after the response has
been sent: it adds no latency, and a failed write can't fail the request.
Duplicates make no model call, so they leave no trace.

GET /stats/llm aggregates those traces (see app/llm_stats.py).
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import PyMongoError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.dedup import find_active_incident, record_duplicate, create_new_incident
from app.llm_client import LLMResponseError, analyze_incident
from app.llm_stats import LLMStats, StatsSource, llm_stats
from app.mongo import create_client, ensure_indexes, get_llm_runs, llm_runs_collection, save_llm_run
from app.parser import compute_error_hash, normalize_raw_text
from app.schemas import IncidentAnalysis, IncidentRecord, RawLogEntry


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Created here, inside the server's event loop, which an async Mongo
    # client is bound to; closed on shutdown.
    mongo_client = create_client()
    app.state.llm_runs = llm_runs_collection(mongo_client)
    await ensure_indexes(app.state.llm_runs)
    yield
    await mongo_client.close()


app = FastAPI(title="AI Log Incident Analyzer", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/stats/llm", response_model=LLMStats)
async def llm_stats_endpoint(days: int = Query(7, ge=1, le=365, description="How many days back to look"),
                             source: StatsSource = Query("api", description="api, eval, or all"),
                             llm_runs: AsyncCollection = Depends(get_llm_runs)) -> LLMStats:
    # Unlike saving a trace, this can't be best effort: without Mongo there
    # is nothing to report, so the client gets a clear 503.
    try:
        return await llm_stats(llm_runs, days=days, source=source)
    except PyMongoError as e:
        # The driver's message names internal hosts and timeouts - log it,
        # don't hand it to the client.
        print(f"[stats] aggregation failed: {e}")
        raise HTTPException(status_code=503, detail="LLM trace store unavailable")


@app.post("/analyze-incident", response_model=IncidentRecord)
async def analyze_incident_endpoint(entry: RawLogEntry,
                                    background_tasks: BackgroundTasks,
                                    session: AsyncSession = Depends(get_session),
                                    llm_runs: AsyncCollection = Depends(get_llm_runs)) -> IncidentRecord:
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
        # .value, not the enum: since Python 3.12 an f-string renders a
        # (str, Enum) member as "Environment.PROD", which is what the prompt
        # used to receive (caught by reading a trace in llm_runs).
        analysis, llm_run = await run_in_threadpool(
            analyze_incident, entry.service, entry.environment.value, normalized_text
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
        llm_latency_ms=llm_run.latency_ms,
        llm_retry_count=llm_run.retry_count,
    )

    # After the Postgres insert, so the trace can point at the incident. There
    # is no transaction spanning both databases: if this write fails, the
    # incident simply has no trace - the reverse (a trace pointing at an
    # incident that was never saved) would be worse.
    background_tasks.add_task(save_llm_run,
                              llm_runs,
                              llm_run,
                              source="api",
                              incident_id=incident.id,
                              error_hash=error_hash,
                              service=entry.service,
                              environment=entry.environment.value)

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
