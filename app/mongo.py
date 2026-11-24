"""
MongoDB layer - where the LLM call traces (app/llm_trace.py) are kept.

Postgres holds the incidents: they need constraints, database-level enum
validation and the indexed dedup lookup. The traces are a different kind of
data - nested, varying in shape (one step or two, one to three attempts
each), append-only and short-lived - so they go to a document store, one
document per analysis in the `llm_runs` collection.

Saving a trace is best effort: it's diagnostics, not business data, so a
Mongo outage must never fail an analysis that already succeeded. Errors are
caught and printed, and the client gives up quickly instead of after
pymongo's default 30 s server-selection timeout.
"""

import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from fastapi import Request
from pymongo import AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import PyMongoError

from app.llm_trace import LLMRun

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://root:root@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "incident_analyzer")

TraceSource = Literal["api", "eval"]


def create_client() -> AsyncMongoClient:
    """
    Builds the client; it connects lazily, on first use. Create it inside
    the running event loop (the FastAPI lifespan does) - an async client
    belongs to the loop it was created in.
    """
    return AsyncMongoClient(
        MONGO_URL,
        # Store uuid.UUID as the standard BSON UUID subtype. pymongo has no
        # default here and refuses to encode a UUID without one.
        uuidRepresentation="standard",
        # Return datetimes as timezone-aware UTC, matching the rest of the app.
        tz_aware=True,
        serverSelectionTimeoutMS=2000,
    )


def llm_runs_collection(client: AsyncMongoClient) -> AsyncCollection:
    return client[MONGO_DB]["llm_runs"]


def get_llm_runs(request: Request) -> AsyncCollection:
    """FastAPI dependency - the collection opened in the app's lifespan."""
    return request.app.state.llm_runs


async def save_llm_run(
    collection: AsyncCollection,
    run: LLMRun,
    *,
    source: TraceSource,
    incident_id: uuid.UUID | None,
    error_hash: str,
    service: str,
    environment: str,
) -> None:
    """
    Inserts one trace document: the run itself plus the context that links
    it back to an incident (incident_id is None for eval runs, which have no
    incident). Never raises on a database error - see module docstring.
    """
    document = {
        "source": source,
        "incident_id": incident_id,
        "error_hash": error_hash,
        "service": service,
        "environment": environment,
        # mode="python" keeps created_at a datetime, stored as a BSON date
        # (which a TTL index needs), rather than an ISO string.
        **run.model_dump(mode="python"),
    }
    try:
        await collection.insert_one(document)
    except PyMongoError as e:
        print(f"[mongo] could not save LLM trace for {service} ({error_hash[:12]}): {e}")
