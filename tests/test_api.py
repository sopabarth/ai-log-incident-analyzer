"""
The HTTP layer with every collaborator faked: Postgres (dedup + insert), the
model and the Mongo trace store. These tests check the endpoint's wiring -
what gets called, with what, in which case - not the collaborators, which
have tests of their own.

TestClient is used without `with`, so the app's lifespan (which connects to
MongoDB) doesn't run; the dependencies that would need it are overridden.
"""

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pymongo.errors import ServerSelectionTimeoutError

import app.main as main
from app.db import get_session
from app.llm_trace import AttemptTrace, LLMRun, StepTrace
from app.mongo import get_llm_runs
from app.schemas import ErrorCategory, IncidentAnalysis, Priority

TRACE_COLLECTION = object()  # stands in for the llm_runs collection

ANALYSIS = IncidentAnalysis(category=ErrorCategory.DATABASE_TIMEOUT, root_cause_summary="Pool exhausted.",
                            priority=Priority.CRITICAL, priority_reasoning="All checkout requests fail.",
                            confidence=0.95, needs_human_review=False)

REQUEST = {"service": "checkout-api", "environment": "prod", "timestamp": "2026-11-24T19:00:00Z",
           "raw_text": "TimeoutError: QueuePool limit reached\n  File \"/app/orders.py\", line 88, in create"}


def an_llm_run() -> LLMRun:
    return LLMRun(mode="single", model="m", steps=[
        StepTrace(name="analyze", system_prompt_fingerprint="abc", user_prompt="u", latency_ms=1234,
                  attempts=[AttemptTrace(n=1, status="invalid_json", latency_ms=600),
                            AttemptTrace(n=2, status="ok", latency_ms=634)])])


def stored_incident(**overrides) -> SimpleNamespace:
    fields = dict(id=uuid.uuid4(), service="checkout-api", environment="prod",
                  timestamp=datetime(2026, 11, 24, 19, tzinfo=timezone.utc), raw_text_hash="h" * 64,
                  occurrence_count=1, llm_retry_count=1, llm_latency_ms=1234,
                  **ANALYSIS.model_dump())
    return SimpleNamespace(**{**fields, **overrides})


class Fakes:
    """Records every call the endpoint makes to its collaborators."""

    def __init__(self):
        self.existing = None  # what find_active_incident returns
        self.analyze_calls: list[tuple] = []
        self.created: list[dict] = []
        self.saved_traces: list[tuple] = []
        self.stats_error: Exception | None = None
        self.stats_calls: list[dict] = []

    async def find_active_incident(self, session, error_hash):
        return self.existing

    async def record_duplicate(self, session, incident):
        incident.occurrence_count += 1
        return incident

    def analyze_incident(self, service, environment, normalized_text):
        self.analyze_calls.append((service, environment, normalized_text))
        return ANALYSIS, an_llm_run()

    async def create_new_incident(self, session, **fields):
        self.created.append(fields)
        return stored_incident(raw_text_hash=fields["raw_text_hash"], llm_retry_count=fields["llm_retry_count"],
                               llm_latency_ms=fields["llm_latency_ms"])

    async def save_llm_run(self, collection, run, **context):
        self.saved_traces.append((collection, run, context))

    async def llm_stats(self, collection, *, days, source):
        self.stats_calls.append({"days": days, "source": source})
        if self.stats_error:
            raise self.stats_error
        return main.LLMStats(since=datetime.now(timezone.utc), source=source, overview=None,
                             by_mode_model=[], steps=[])


@pytest.fixture
def fakes(monkeypatch) -> Fakes:
    f = Fakes()
    for name in ("find_active_incident", "record_duplicate", "analyze_incident", "create_new_incident",
                 "save_llm_run", "llm_stats"):
        monkeypatch.setattr(main, name, getattr(f, name))
    return f


@pytest.fixture
def client(fakes) -> TestClient:
    async def no_session():
        yield None

    main.app.dependency_overrides[get_session] = no_session
    main.app.dependency_overrides[get_llm_runs] = lambda: TRACE_COLLECTION
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


# --- POST /analyze-incident ---

def test_new_incident_is_analyzed_stored_and_traced(client, fakes):
    response = client.post("/analyze-incident", json=REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["is_duplicate"] is False and body["duplicate_of_id"] is None
    assert body["analysis"]["category"] == "database_timeout"

    # The model gets the normalized text and the environment as a plain
    # string - "prod", not "Environment.PROD" (a bug the traces caught).
    # Equality alone can't catch that bug: Environment.PROD == "prod" is
    # True, since the enum subclasses str. What breaks is formatting it.
    [(service, environment, text)] = fakes.analyze_calls
    assert service == "checkout-api"
    assert type(environment) is str and f"{environment}" == "prod"
    assert text == 'TimeoutError: QueuePool limit reached\nFile "/app/orders.py", line 88, in create'

    # Retry count and latency come from the trace.
    [created] = fakes.created
    assert created["llm_retry_count"] == 1 and created["llm_latency_ms"] == 1234
    assert body["llm_retry_count"] == 1 and body["llm_latency_ms"] == 1234

    # The trace is saved (as a background task) and points at the new incident.
    [(collection, run, context)] = fakes.saved_traces
    assert collection is TRACE_COLLECTION
    assert run.retry_count == 1
    assert context == {"incident_id": uuid.UUID(body["id"]), "error_hash": body["raw_text_hash"],
                       "service": "checkout-api", "environment": "prod"}


def test_environment_alias_reaches_the_model_as_the_canonical_value(client, fakes):
    assert client.post("/analyze-incident", json={**REQUEST, "environment": "production"}).status_code == 200
    assert f"{fakes.analyze_calls[0][1]}" == "prod"


def test_duplicate_skips_the_model_and_leaves_no_trace(client, fakes):
    fakes.existing = stored_incident()
    response = client.post("/analyze-incident", json=REQUEST)

    assert response.status_code == 200
    body = response.json()
    assert body["is_duplicate"] is True
    assert body["duplicate_of_id"] == str(fakes.existing.id)
    assert body["occurrence_count"] == 2
    assert fakes.analyze_calls == [] and fakes.created == [] and fakes.saved_traces == []


def test_blank_raw_text_is_rejected_before_anything_runs(client, fakes):
    response = client.post("/analyze-incident", json={**REQUEST, "raw_text": "  \n\t\n"})
    assert response.status_code == 400
    assert fakes.analyze_calls == [] and fakes.created == []


@pytest.mark.parametrize("change", [{"environment": "qa"}, {"timestamp": "yesterday"}, {"raw_text": None}])
def test_invalid_request_is_a_422(client, fakes, change):
    assert client.post("/analyze-incident", json={**REQUEST, **change}).status_code == 422
    assert fakes.analyze_calls == []


# --- GET /stats/llm ---

def test_stats_defaults_and_parameters(client, fakes):
    assert client.get("/stats/llm").status_code == 200
    assert client.get("/stats/llm?days=30&source=all").status_code == 200
    assert fakes.stats_calls == [{"days": 7, "source": "api"}, {"days": 30, "source": "all"}]


@pytest.mark.parametrize("query", ["days=0", "days=366", "source=prod"])
def test_stats_rejects_bad_parameters(client, fakes, query):
    assert client.get(f"/stats/llm?{query}").status_code == 422
    assert fakes.stats_calls == []


def test_stats_is_a_503_without_internal_details_when_mongo_is_down(client, fakes):
    fakes.stats_error = ServerSelectionTimeoutError("mongo:27017: [Errno 104] Connection reset by peer")
    response = client.get("/stats/llm")
    assert response.status_code == 503
    assert response.json() == {"detail": "LLM trace store unavailable"}
