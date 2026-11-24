"""
Statistics over the LLM call traces in llm_runs, computed by MongoDB itself
with one aggregation pipeline.

A pipeline is a list of stages, each transforming the stream of documents
the previous one produced - like a chain of SQL clauses, but in an explicit
order:

  $match    filter documents (WHERE); first, so it can use an index
  $facet    run several sub-pipelines over the same matched documents and
            return all their results in one document - one round trip
            instead of three queries
  $unwind   turn one document with an array into one document per element,
            so nested data (steps, and attempts inside steps) can be grouped
  $group    aggregate (GROUP BY), with accumulators such as $sum, $avg,
            $max and $percentile
  $project  shape the output (SELECT ... AS ...)

build_stats_pipeline() only builds the pipeline (plain data, easy to read
and print); llm_stats() runs it.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

from pydantic import BaseModel
from pymongo.asynchronous.collection import AsyncCollection

StatsSource = Literal["api", "eval", "all"]


class LatencyStats(BaseModel):
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: int


class OverviewStats(BaseModel):
    runs: int
    fallbacks: int
    fallback_rate: float
    retries: int
    latency: LatencyStats


class ModeModelStats(BaseModel):
    mode: str
    model: str
    runs: int
    fallback_rate: float
    mean_retry_count: float
    latency: LatencyStats


class StepStats(BaseModel):
    step: str
    attempts: int
    failed: int
    failure_rate: float
    statuses: dict[str, int]  # attempt status -> how many attempts ended that way


class LLMStats(BaseModel):
    since: datetime
    source: StatsSource
    overview: OverviewStats | None  # None when no trace matched
    by_mode_model: list[ModeModelStats]
    steps: list[StepStats]


# Accumulators and projection shared by the overview and per-mode groups.
# $percentile (MongoDB 7.0+) returns an array, one value per requested p;
# "approximate" is the only method $group supports.
_LATENCY_ACCUMULATORS = {
    "mean_latency": {"$avg": "$latency_ms"},
    "max_latency": {"$max": "$latency_ms"},
    "latency_percentiles": {"$percentile": {"input": "$latency_ms", "p": [0.5, 0.95], "method": "approximate"}},
}
_LATENCY_PROJECTION = {
    "mean_ms": {"$round": ["$mean_latency", 1]},
    "p50_ms": {"$arrayElemAt": ["$latency_percentiles", 0]},
    "p95_ms": {"$arrayElemAt": ["$latency_percentiles", 1]},
    "max_ms": "$max_latency",
}


def build_stats_pipeline(since: datetime, source: StatsSource) -> list[dict]:
    match: dict = {"created_at": {"$gte": since}}
    if source != "all":
        match["source"] = source

    overview = [
        {"$group": {
            "_id": None,
            "runs": {"$sum": 1},
            "fallbacks": {"$sum": {"$cond": ["$fell_back", 1, 0]}},
            "retries": {"$sum": "$retry_count"},
            **_LATENCY_ACCUMULATORS,
        }},
        {"$project": {
            "_id": 0,
            "runs": 1,
            "fallbacks": 1,
            "fallback_rate": {"$round": [{"$divide": ["$fallbacks", "$runs"]}, 3]},
            "retries": 1,
            "latency": _LATENCY_PROJECTION,
        }},
    ]

    by_mode_model = [
        {"$group": {
            # A compound _id groups by both fields, like GROUP BY mode, model.
            "_id": {"mode": "$mode", "model": "$model"},
            "runs": {"$sum": 1},
            "fallbacks": {"$sum": {"$cond": ["$fell_back", 1, 0]}},
            "mean_retry_count": {"$avg": "$retry_count"},
            **_LATENCY_ACCUMULATORS,
        }},
        {"$sort": {"_id.mode": 1, "_id.model": 1}},
        {"$project": {
            "_id": 0,
            "mode": "$_id.mode",
            "model": "$_id.model",
            "runs": 1,
            "fallback_rate": {"$round": [{"$divide": ["$fallbacks", "$runs"]}, 3]},
            "mean_retry_count": {"$round": ["$mean_retry_count", 2]},
            "latency": _LATENCY_PROJECTION,
        }},
    ]

    steps = [
        # One document per step, then one per attempt within it: a run with
        # two steps of 2 and 1 attempts becomes three documents.
        {"$unwind": "$steps"},
        {"$unwind": "$steps.attempts"},
        # Count attempts per (step, status)...
        {"$group": {
            "_id": {"step": "$steps.name", "status": "$steps.attempts.status"},
            "count": {"$sum": 1},
        }},
        # ...then roll those counts up per step, collecting the statuses as
        # {k, v} pairs that $arrayToObject turns into {"ok": 5, ...}.
        {"$group": {
            "_id": "$_id.step",
            "attempts": {"$sum": "$count"},
            "failed": {"$sum": {"$cond": [{"$eq": ["$_id.status", "ok"]}, 0, "$count"]}},
            "statuses": {"$push": {"k": "$_id.status", "v": "$count"}},
        }},
        {"$sort": {"_id": 1}},
        {"$project": {
            "_id": 0,
            "step": "$_id",
            "attempts": 1,
            "failed": 1,
            "failure_rate": {"$round": [{"$divide": ["$failed", "$attempts"]}, 3]},
            "statuses": {"$arrayToObject": "$statuses"},
        }},
    ]

    return [
        # Before $facet, so it can use an index (the TTL index on created_at
        # doubles as an ordinary one for this range); stages inside $facet
        # never use indexes.
        {"$match": match},
        {"$facet": {"overview": overview, "by_mode_model": by_mode_model, "steps": steps}},
    ]


async def llm_stats(collection: AsyncCollection, *, days: int, source: StatsSource) -> LLMStats:
    """Stats over the traces of the last `days` days. Raises PyMongoError if Mongo is unreachable."""
    since = datetime.now(timezone.utc) - timedelta(days=days)
    cursor = await collection.aggregate(build_stats_pipeline(since, source))
    # $facet always yields exactly one document, holding one array per facet.
    [result] = await cursor.to_list()
    overview = result["overview"]
    return LLMStats(since=since,
                    source=source,
                    overview=overview[0] if overview else None,
                    by_mode_model=result["by_mode_model"],
                    steps=result["steps"])
