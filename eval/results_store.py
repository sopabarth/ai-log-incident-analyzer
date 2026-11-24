"""
Eval results in MongoDB - replaces the JSON files eval/run_eval.py used to
write under eval/results/.

Two collections, one per kind of data:

  eval_runs  one document per (eval invocation, mode): which model and
             dataset, the computed metrics, and every example's result
             EMBEDDED in it. Embedding fits because the examples belong to
             exactly one eval run, are always read with it, and are bounded
             (41 small entries, far below the 16 MB document limit).

  llm_runs   the same collection the API writes, with source="eval": the
             full trace of every example's model calls. These are
             REFERENCED instead - each embedded example holds the trace's
             _id (trace_id), and each trace holds eval_run_id - because they
             are large, mostly not needed, and expire with the TTL index
             (eval_runs documents don't: results are kept for comparison
             over time, the raw traces only for LLM_TRACE_TTL_DAYS).

Writes are ordered like the API's: the traces first, then the eval_runs
document that points at them. A failure in between leaves traces without
an eval run (which expire), never an eval run pointing at missing traces.

Unlike the API, the eval does NOT swallow Mongo errors: whoever ran it with
--save asked for the results to be stored, so a failure must be loud.
"""

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId

from app.llm_trace import LLMRun
from app.mongo import MONGO_DB, create_client, ensure_indexes, llm_runs_collection, trace_document

EVAL_RUNS = "eval_runs"


def dataset_fingerprint(path: Path) -> str:
    """Hash of the dataset file. Results are only comparable across runs on
    the same dataset - an edited label changes every metric."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


async def save_eval_run(*, batch_id: ObjectId, mode: str, model: str, dataset_path: Path, examples_total: int,
                        limit: int | None, metrics: dict, results: list, traces: list[LLMRun]) -> ObjectId:
    """
    Stores one mode's eval run: every example's trace in llm_runs, then one
    eval_runs document. `results` (ExampleResult dataclasses) and `traces`
    are parallel lists, one entry per example. batch_id is shared by the
    modes of one invocation (`--mode both`), so they can be paired later.
    Returns the eval run's _id.
    """
    client = create_client()
    try:
        llm_runs = llm_runs_collection(client)
        await ensure_indexes(llm_runs)

        # The eval run's _id is generated here, before the document exists,
        # so the traces (inserted first) can already point at it. ObjectIds
        # are made client-side anyway; that's what makes this possible.
        eval_run_id = ObjectId()
        trace_docs = [trace_document(trace,
                                     source="eval",
                                     eval_run_id=eval_run_id,
                                     error_hash=result.error_hash,
                                     service=result.service,
                                     environment=result.environment)
                      for result, trace in zip(results, traces, strict=True)]
        # One round trip for all traces; inserted_ids come back in order.
        inserted = await llm_runs.insert_many(trace_docs)

        await client[MONGO_DB][EVAL_RUNS].insert_one({
            "_id": eval_run_id,
            "batch_id": batch_id,
            "created_at": datetime.now(timezone.utc),
            "mode": mode,
            "model": model,
            # The file name, not the path: an absolute path is specific to
            # one machine, and the fingerprint is what identifies the data.
            "dataset": {"file": dataset_path.name,
                        "sha256": dataset_fingerprint(dataset_path),
                        "examples": examples_total},
            # None means the whole dataset; only complete runs are comparable.
            "limit": limit,
            "metrics": metrics,
            "examples": [{**asdict(result), "trace_id": trace_id}
                         for result, trace_id in zip(results, inserted.inserted_ids, strict=True)],
        })
        return eval_run_id
    finally:
        await client.close()


def history_pipeline(dataset_sha256: str) -> list[dict]:
    """
    Every complete eval run on this dataset, summarized per (mode, model).
    A single run on 41 examples is noisy - one example is 2.4 points - so
    the spread across runs ($stdDevSamp, $min, $max) is what tells whether
    a difference between modes is real.
    """
    def spread(field: str) -> dict:
        return {"mean": {"$avg": field}, "stddev": {"$stdDevSamp": field}, "min": {"$min": field}, "max": {"$max": field}}

    category = spread("$metrics.category.accuracy")
    priority = spread("$metrics.priority.exact_match_accuracy")
    return [
        {"$match": {"dataset.sha256": dataset_sha256, "limit": None}},
        {"$group": {
            "_id": {"mode": "$mode", "model": "$model"},
            "runs": {"$sum": 1},
            "last_run": {"$max": "$created_at"},
            **{f"category_{k}": v for k, v in category.items()},
            **{f"priority_{k}": v for k, v in priority.items()},
            "mean_distance": {"$avg": "$metrics.priority.mean_distance"},
            "median_latency_ms": {"$avg": "$metrics.latency.median_latency_ms"},
        }},
        {"$sort": {"_id.model": 1, "_id.mode": 1}},
    ]


async def load_history(dataset_path: Path, recent: int = 10) -> tuple[list[dict], list[dict]]:
    """(the most recent eval runs, newest first; the per-mode summary)."""
    client = create_client()
    try:
        eval_runs = client[MONGO_DB][EVAL_RUNS]
        # A plain find is enough for "latest N": filter, sort, limit, and a
        # projection so the 41 embedded examples aren't sent over the wire.
        cursor = eval_runs.find({}, {"examples": 0}).sort("created_at", -1).limit(recent)
        recent_runs = await cursor.to_list()
        summary_cursor = await eval_runs.aggregate(history_pipeline(dataset_fingerprint(dataset_path)))
        return recent_runs, await summary_cursor.to_list()
    finally:
        await client.close()
