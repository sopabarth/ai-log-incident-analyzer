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

Indexes play the role migrations play for Postgres, but MongoDB has no
schema to migrate: ensure_indexes() runs at app startup and declares the
indexes the collection should have. Creating an index that already exists
with the same options is a no-op, so this is safe on every start.
"""

import os
import uuid
from typing import Literal

from dotenv import load_dotenv
from fastapi import Request
from pymongo import ASCENDING, DESCENDING, AsyncMongoClient
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.errors import PyMongoError

from app.llm_trace import LLMRun

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://root:root@localhost:27017/")
MONGO_DB = os.getenv("MONGO_DB", "incident_analyzer")
# Traces hold raw log text (possibly secrets or personal data from a stack
# trace) and lose their value with age, so they expire.
TRACE_TTL_DAYS = int(os.getenv("LLM_TRACE_TTL_DAYS", 30))

_TTL_INDEX = "created_at_ttl"

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


async def ensure_indexes(collection: AsyncCollection, ttl_seconds: int = TRACE_TTL_DAYS * 86400) -> None:
    """
    Creates the collection's indexes if missing, and brings the TTL up to
    date if LLM_TRACE_TTL_DAYS changed. Best effort, like saving: with Mongo
    down the app still starts, and the indexes are created on the next start.
    """
    try:
        # Look up the trace of one incident. Unique, because an incident is
        # analyzed once - a second trace for the same incident would be a bug.
        # Partial, because eval traces have no incident: they leave the field
        # out, and a plain unique index would treat every missing value as the
        # same null and allow only one such trace.
        #
        # The filter is $exists rather than {$type: "binData"}: the planner
        # only uses a partial index when it can prove the query falls inside
        # the filter, and it proves {incident_id: <uuid>} implies $exists but
        # not $type - with $type, lookups by incident_id were full collection
        # scans. $exists counts an explicit null as present, which is why eval
        # traces omit the field instead of storing null.
        await collection.create_index("incident_id",
                                      name="incident_id_unique",
                                      unique=True,
                                      partialFilterExpression={"incident_id": {"$exists": True}})
        # All traces of one error, newest first. A compound index serves both
        # the filter and the sort, so Mongo reads only the matching entries
        # in order instead of sorting in memory.
        await collection.create_index([("error_hash", ASCENDING), ("created_at", DESCENDING)],
                                      name="error_hash_created_at")
        await _ensure_ttl_index(collection, ttl_seconds)
    except PyMongoError as e:
        print(f"[mongo] could not ensure indexes on {collection.name}: {e}")


async def _ensure_ttl_index(collection: AsyncCollection, ttl_seconds: int) -> None:
    """
    A TTL index makes Mongo's background monitor (it runs every 60 s) delete
    documents whose created_at is older than ttl_seconds. It only works on a
    BSON date field, which is why created_at is stored as a datetime.

    create_index() can't change an existing index's options - it fails with
    IndexOptionsConflict - so a changed TTL is applied with collMod instead,
    in place, without dropping and rebuilding the index.
    """
    existing = (await collection.index_information()).get(_TTL_INDEX)
    if existing is None:
        await collection.create_index("created_at", name=_TTL_INDEX, expireAfterSeconds=ttl_seconds)
    elif existing.get("expireAfterSeconds") != ttl_seconds:
        await collection.database.command("collMod", collection.name,
                                          index={"name": _TTL_INDEX, "expireAfterSeconds": ttl_seconds})
        print(f"[mongo] {collection.name}: TTL changed from {existing.get('expireAfterSeconds')}s to {ttl_seconds}s")


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
    it back to an incident. Eval runs have no incident (incident_id=None);
    their documents leave the field out rather than storing null, which the
    unique incident_id index relies on (see ensure_indexes). Never raises on
    a database error - see module docstring.
    """
    document = {
        "source": source,
        **({"incident_id": incident_id} if incident_id is not None else {}),
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
