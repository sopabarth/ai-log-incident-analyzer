"""
The parts of app/mongo.py that need no real MongoDB: how a trace document
is built, and that saving one never fails a request. Indexes and queries
against a real server are in test_mongo_integration.py.
"""

import uuid

import pytest
from bson import ObjectId
from pymongo.errors import ServerSelectionTimeoutError

from app.llm_trace import AttemptTrace, LLMRun, StepTrace
from app.mongo import save_llm_run, trace_document


def a_run() -> LLMRun:
    return LLMRun(mode="single", model="m", steps=[
        StepTrace(name="analyze", system_prompt_fingerprint="abc", user_prompt="u", latency_ms=800,
                  attempts=[AttemptTrace(n=1, status="ok", latency_ms=800, raw_response="{}")])])


def test_api_trace_document_links_to_the_incident():
    incident_id = uuid.uuid4()
    doc = trace_document(a_run(), source="api", incident_id=incident_id, error_hash="h", service="s",
                         environment="prod")
    assert doc["source"] == "api" and doc["incident_id"] == incident_id
    assert "eval_run_id" not in doc
    assert doc["mode"] == "single" and doc["latency_ms"] == 800 and doc["steps"][0]["name"] == "analyze"


def test_eval_trace_document_leaves_incident_id_out_instead_of_null():
    # The unique incident_id index counts null as a value, so an eval trace
    # storing incident_id: null would collide with the next one.
    eval_run_id = ObjectId()
    doc = trace_document(a_run(), source="eval", eval_run_id=eval_run_id, error_hash="h", service="s",
                         environment="dev")
    assert "incident_id" not in doc
    assert doc["eval_run_id"] == eval_run_id


class RecordingCollection:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.inserted: list[dict] = []

    async def insert_one(self, document: dict):
        if self.error:
            raise self.error
        self.inserted.append(document)


@pytest.mark.anyio
async def test_save_llm_run_inserts_an_api_trace():
    collection = RecordingCollection()
    incident_id = uuid.uuid4()
    await save_llm_run(collection, a_run(), incident_id=incident_id, error_hash="h" * 64, service="s",
                       environment="prod")
    [doc] = collection.inserted
    assert doc["source"] == "api" and doc["incident_id"] == incident_id


@pytest.mark.anyio
async def test_save_llm_run_never_raises_when_mongo_is_down(capsys):
    collection = RecordingCollection(error=ServerSelectionTimeoutError("No servers found yet"))
    await save_llm_run(collection, a_run(), incident_id=uuid.uuid4(), error_hash="ab" * 32, service="checkout",
                       environment="prod")
    assert "could not save LLM trace for checkout (abababababab)" in capsys.readouterr().out
