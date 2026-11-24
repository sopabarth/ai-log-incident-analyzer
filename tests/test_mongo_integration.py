"""
Against a real MongoDB: the indexes, the stats pipeline and the eval store
are logic that runs inside the database, which no fake can check.

Skipped unless MONGO_TEST_URL is set, e.g.

    MONGO_TEST_URL=mongodb://root:root@localhost:27017/ uv run pytest

Each test gets its own throwaway database (dropped afterwards), so it can
safely point at the development server.
"""

import os
import uuid
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

import app.mongo as mongo
import eval.results_store as results_store
from app.llm_stats import llm_stats
from app.llm_trace import AttemptTrace, LLMRun, StepTrace
from app.mongo import create_client, ensure_indexes, llm_runs_collection, trace_document
from eval.run_eval import ExampleResult

MONGO_TEST_URL = os.getenv("MONGO_TEST_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.anyio,
    pytest.mark.skipif(not MONGO_TEST_URL, reason="MONGO_TEST_URL is not set"),
]


@pytest.fixture(autouse=True)
def test_database(monkeypatch):
    """Points app.mongo (and the eval store) at a fresh database for this test."""
    name = f"incident_analyzer_test_{uuid.uuid4().hex[:8]}"
    monkeypatch.setattr(mongo, "MONGO_URL", MONGO_TEST_URL)
    monkeypatch.setattr(mongo, "MONGO_DB", name)
    monkeypatch.setattr(results_store, "MONGO_DB", name)
    yield name
    with MongoClient(MONGO_TEST_URL) as client:
        client.drop_database(name)


def run_doc(mode: str, latency_ms: int, steps: list[tuple[str, list[str]]], *, fell_back: bool = False,
            source: str = "api", age: timedelta = timedelta(hours=1)) -> dict:
    """A trace as stored, with full control over the numbers the stats aggregate."""
    retry_count = sum(len(statuses) - 1 for _, statuses in steps)
    return {"source": source, "mode": mode, "model": "m", "latency_ms": latency_ms, "retry_count": retry_count,
            "fell_back": fell_back, "created_at": datetime.now(timezone.utc) - age,
            "steps": [{"name": name, "attempts": [{"n": i + 1, "status": s} for i, s in enumerate(statuses)]}
                      for name, statuses in steps]}


def an_llm_run(latency_ms: int = 900) -> LLMRun:
    return LLMRun(mode="single", model="m", steps=[
        StepTrace(name="analyze", system_prompt_fingerprint="abc", user_prompt="u", latency_ms=latency_ms,
                  attempts=[AttemptTrace(n=1, status="ok", latency_ms=latency_ms, raw_response='{"ok": 1}')])])


# --- indexes ---

async def test_ensure_indexes_is_idempotent_and_updates_the_ttl():
    client = create_client()
    try:
        runs = llm_runs_collection(client)
        await ensure_indexes(runs, ttl_seconds=3600)
        await ensure_indexes(runs, ttl_seconds=3600)  # a second start changes nothing
        indexes = await runs.index_information()
        assert set(indexes) == {"_id_", "incident_id_unique", "error_hash_created_at", "created_at_ttl"}
        assert indexes["incident_id_unique"]["unique"] is True
        assert indexes["created_at_ttl"]["expireAfterSeconds"] == 3600

        await ensure_indexes(runs, ttl_seconds=60)  # LLM_TRACE_TTL_DAYS changed
        assert (await runs.index_information())["created_at_ttl"]["expireAfterSeconds"] == 60
    finally:
        await client.close()


async def test_one_trace_per_incident_but_any_number_of_eval_traces():
    client = create_client()
    try:
        runs = llm_runs_collection(client)
        await ensure_indexes(runs)
        incident_id = uuid.uuid4()
        api = dict(source="api", incident_id=incident_id, error_hash="h", service="s", environment="prod")
        await runs.insert_one(trace_document(an_llm_run(), **api))
        with pytest.raises(DuplicateKeyError):
            await runs.insert_one(trace_document(an_llm_run(), **api))

        eval_run_id = ObjectId()
        await runs.insert_many([trace_document(an_llm_run(), source="eval", eval_run_id=eval_run_id,
                                               error_hash="h", service="s", environment="dev") for _ in range(3)])
        assert await runs.count_documents({"eval_run_id": eval_run_id}) == 3

        # The partial index is used for lookups by incident (it wasn't with a
        # $type filter - see ensure_indexes).
        plan = await runs.find({"incident_id": incident_id}).explain()
        assert "incident_id_unique" in str(plan["queryPlanner"]["winningPlan"])
    finally:
        await client.close()


# --- stats pipeline ---

async def test_llm_stats_against_hand_computed_values():
    client = create_client()
    try:
        runs = llm_runs_collection(client)
        await ensure_indexes(runs)
        await runs.insert_many([
            run_doc("single", 1000, [("analyze", ["ok"])]),
            run_doc("single", 3000, [("analyze", ["invalid_json", "ok"])]),
            run_doc("single", 9000, [("analyze", ["rate_limited", "timeout", "validation_error"])], fell_back=True),
            run_doc("decomposed", 2000, [("classify", ["validation_error", "ok"]), ("prioritize", ["ok"])]),
            run_doc("decomposed", 4000, [("classify", ["ok"]), ("prioritize", ["ok"])]),
            # Excluded from the default query: too old, and not real traffic.
            run_doc("single", 99999, [("analyze", ["ok"])], age=timedelta(days=30)),
            run_doc("single", 88888, [("analyze", ["ok"])], source="eval"),
        ])

        stats = await llm_stats(runs, days=7, source="api")

        o = stats.overview
        assert (o.runs, o.fallbacks, o.fallback_rate, o.retries) == (5, 1, 0.2, 4)
        assert (o.latency.mean_ms, o.latency.p50_ms, o.latency.max_ms) == (3800.0, 3000.0, 9000)

        decomposed, single = stats.by_mode_model  # sorted by mode
        assert (decomposed.mode, decomposed.runs, decomposed.mean_retry_count, decomposed.fallback_rate) == \
               ("decomposed", 2, 0.5, 0.0)
        assert (single.mode, single.runs, single.mean_retry_count, single.fallback_rate) == ("single", 3, 1.0, 0.333)
        assert single.latency.mean_ms == 4333.3

        steps = {s.step: s for s in stats.steps}
        assert (steps["analyze"].attempts, steps["analyze"].failed, steps["analyze"].failure_rate) == (6, 4, 0.667)
        assert steps["analyze"].statuses == {"ok": 2, "invalid_json": 1, "rate_limited": 1, "timeout": 1,
                                             "validation_error": 1}
        assert steps["classify"].statuses == {"ok": 2, "validation_error": 1}
        assert steps["prioritize"].failed == 0

        assert (await llm_stats(runs, days=7, source="all")).overview.runs == 6
        assert (await llm_stats(runs, days=60, source="api")).overview.runs == 6
        assert (await llm_stats(runs, days=7, source="eval")).overview.runs == 1
    finally:
        await client.close()


async def test_llm_stats_with_no_matching_traces():
    client = create_client()
    try:
        stats = await llm_stats(llm_runs_collection(client), days=7, source="api")
        assert stats.overview is None and stats.by_mode_model == [] and stats.steps == []
    finally:
        await client.close()


# --- eval store ---

def example_result(service: str) -> ExampleResult:
    return ExampleResult(service=service, environment="prod", expected_category="unknown",
                         actual_category="unknown", expected_priority="low", actual_priority="low",
                         expected_needs_human_review=False, actual_needs_human_review=False, confidence=0.9,
                         latency_ms=900, retry_count=0, fell_back=False, error_hash="h", notes="")


def metrics(category: float, priority: float) -> dict:
    return {"category": {"accuracy": category}, "needs_human_review": {"accuracy": 1.0},
            "priority": {"exact_match_accuracy": priority, "mean_distance": 0.5},
            "latency": {"median_latency_ms": 1000, "fallbacks": 0}}


async def save(dataset: Path, *, mode: str, category: float, priority: float, limit: int | None = None) -> ObjectId:
    results = [example_result("a"), example_result("b")]
    return await results_store.save_eval_run(batch_id=ObjectId(), mode=mode, model="m", dataset_path=dataset,
                                             examples_total=2, limit=limit, metrics=metrics(category, priority),
                                             results=results, traces=[an_llm_run(), an_llm_run()])


async def test_save_eval_run_embeds_results_and_references_traces(tmp_path, test_database):
    dataset = tmp_path / "logs.json"
    dataset.write_text("[]")
    eval_run_id = await save(dataset, mode="single", category=0.9, priority=0.6)

    client = create_client()
    try:
        db = client[test_database]
        run = await db["eval_runs"].find_one({"_id": eval_run_id})
        assert run["dataset"] == {"file": "logs.json", "sha256": results_store.dataset_fingerprint(dataset),
                                  "examples": 2}
        assert [e["service"] for e in run["examples"]] == ["a", "b"]
        assert {k for k in asdict(example_result("a"))} < set(run["examples"][0])

        # Both directions of the reference agree.
        traces = await db["llm_runs"].find({"eval_run_id": eval_run_id}).to_list()
        assert {t["_id"] for t in traces} == {e["trace_id"] for e in run["examples"]}
        assert all(t["source"] == "eval" and "incident_id" not in t for t in traces)

        # ...and $lookup can follow it.
        cursor = await db["eval_runs"].aggregate([
            {"$match": {"_id": eval_run_id}}, {"$unwind": "$examples"},
            {"$lookup": {"from": "llm_runs", "localField": "examples.trace_id", "foreignField": "_id",
                         "as": "trace"}},
            {"$project": {"_id": 0, "n": {"$size": "$trace"}}}])
        assert [d["n"] for d in await cursor.to_list()] == [1, 1]
    finally:
        await client.close()


async def test_history_summarizes_only_comparable_runs(tmp_path):
    dataset = tmp_path / "logs.json"
    dataset.write_text("[]")
    await save(dataset, mode="single", category=0.9, priority=0.6)
    await save(dataset, mode="single", category=1.0, priority=0.7)
    await save(dataset, mode="single", category=0.1, priority=0.1, limit=1)  # partial run: not comparable
    await save(dataset, mode="decomposed", category=1.0, priority=0.8)
    dataset.write_text('[{"edited": true}]')  # a new dataset version
    await save(dataset, mode="single", category=0.2, priority=0.2)
    dataset.write_text("[]")  # back to the version being summarized

    recent, summary = await results_store.load_history(dataset, recent=10)

    assert len(recent) == 5
    assert all("examples" not in r for r in recent)  # projected away
    assert [r["created_at"] for r in recent] == sorted((r["created_at"] for r in recent), reverse=True)

    rows = {row["_id"]["mode"]: row for row in summary}
    assert rows["single"]["runs"] == 2
    assert rows["single"]["category_mean"] == pytest.approx(0.95)
    assert rows["single"]["category_stddev"] == pytest.approx(0.0707, abs=1e-4)
    assert (rows["single"]["priority_min"], rows["single"]["priority_max"]) == (0.6, 0.7)
    assert rows["decomposed"]["runs"] == 1
    assert rows["decomposed"]["category_stddev"] is None  # undefined for one run
