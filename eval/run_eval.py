"""
Scores the real analysis pipeline against data/synthetic_logs.json's hand-
labeled ground truth: category accuracy/precision/recall, priority exact-
match plus "how far off" distance, needs_human_review accuracy, and
latency/retry stats.

Calls normalize_raw_text() + llm_client.analyze_incident() directly - real
Groq calls, no mocking - but deliberately bypasses FastAPI/dedup/Postgres:
this measures model+pipeline quality, not the HTTP/DB plumbing (already
covered by manual testing), and skipping dedup means every example gets a
fresh real call even if two raw texts happen to normalize identically.

Usage:
    python -m eval.run_eval                # uses the current TASK_DECOMPOSITION setting
    python -m eval.run_eval --mode single
    python -m eval.run_eval --mode decomposed
    python -m eval.run_eval --mode both     # run every example through both, compare head-to-head
    python -m eval.run_eval --limit 5       # smoke test on a subset
    python -m eval.run_eval --save          # also store results + traces in MongoDB
    python -m eval.run_eval --history       # past runs and per-mode spread, from MongoDB

--save stores each mode's run in the eval_runs collection, and every
example's model-call trace in llm_runs (source "eval") - see
eval/results_store.py. Without --save (or --history) the eval doesn't touch
MongoDB at all.
"""

import argparse
import asyncio
import json
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from bson import ObjectId
from pymongo.errors import PyMongoError

import app.llm_client as llm_client
from app.llm_trace import LLMRun
from app.parser import compute_error_hash, normalize_raw_text
from eval.results_store import dataset_fingerprint, load_history, save_eval_run

_DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_logs.json"

_PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


@dataclass
class ExampleResult:
    service: str
    environment: str
    expected_category: str
    actual_category: str
    expected_priority: str
    actual_priority: str
    expected_needs_human_review: bool
    actual_needs_human_review: bool
    confidence: float
    latency_ms: int
    retry_count: int
    fell_back: bool
    error_hash: str
    notes: str


def _load_examples(limit: int | None) -> list[dict]:
    examples = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return examples[:limit] if limit else examples


async def _run_one(example: dict) -> tuple[ExampleResult, LLMRun]:
    normalized = normalize_raw_text(example["raw_text"])
    analysis, llm_run = await llm_client.analyze_incident(
        example["service"], example["environment"], normalized
    )
    result = ExampleResult(
        service=example["service"],
        environment=example["environment"],
        expected_category=example["expected_category"],
        actual_category=analysis.category.value,
        expected_priority=example["expected_priority"],
        actual_priority=analysis.priority.value,
        expected_needs_human_review=example["expected_needs_human_review"],
        actual_needs_human_review=analysis.needs_human_review,
        confidence=analysis.confidence,
        latency_ms=llm_run.latency_ms,
        retry_count=llm_run.retry_count,
        fell_back=llm_run.fell_back,
        error_hash=compute_error_hash(normalized),
        notes=example.get("notes", ""),
    )
    return result, llm_run


async def _run_pipeline(examples: list[dict], mode: str) -> tuple[list[ExampleResult], list[LLMRun]]:
    """
    Runs every example with llm_client.TASK_DECOMPOSITION forced to `mode`
    ("single" or "decomposed"), restoring the original value afterwards.
    Overriding the module attribute directly (rather than the env var)
    works because analyze_incident() reads it fresh on every call, not
    once at import time.

    Returns the results and, in a parallel list, the trace of each example's
    model calls.

    Examples run one at a time even though the client is async: the free
    tier's tokens-per-minute limit would only turn a concurrent run into a
    stream of 429s.
    """
    original = llm_client.TASK_DECOMPOSITION
    llm_client.TASK_DECOMPOSITION = mode == "decomposed"
    try:
        results, traces = [], []
        for i, example in enumerate(examples, start=1):
            print(
                f"  [{mode}] {i}/{len(examples)}: {example['service']} "
                f"(expected {example['expected_category']})...",
                end=" ",
                flush=True,
            )
            result, trace = await _run_one(example)
            verdict = "OK" if result.actual_category == result.expected_category else "MISS"
            if result.fell_back:
                verdict = "FALLBACK"
            print(f"{verdict} -> {result.actual_category}/{result.actual_priority}")
            results.append(result)
            traces.append(trace)
        return results, traces
    finally:
        llm_client.TASK_DECOMPOSITION = original


def _category_report(results: list[ExampleResult]) -> dict:
    categories = sorted({r.expected_category for r in results} | {r.actual_category for r in results})
    per_category = {}
    for cat in categories:
        true_positives = sum(1 for r in results if r.actual_category == cat and r.expected_category == cat)
        predicted_count = sum(1 for r in results if r.actual_category == cat)
        actual_count = sum(1 for r in results if r.expected_category == cat)
        per_category[cat] = {
            "precision": true_positives / predicted_count if predicted_count else None,
            "recall": true_positives / actual_count if actual_count else None,
            "support": actual_count,
        }
    correct = sum(1 for r in results if r.actual_category == r.expected_category)
    return {"accuracy": correct / len(results), "per_category": per_category}


def _priority_report(results: list[ExampleResult]) -> dict:
    distances = [abs(_PRIORITY_ORDER[r.actual_priority] - _PRIORITY_ORDER[r.expected_priority]) for r in results]
    distance_distribution: dict[int, int] = defaultdict(int)
    for d in distances:
        distance_distribution[d] += 1
    exact = sum(1 for d in distances if d == 0)
    return {
        "exact_match_accuracy": exact / len(results),
        "mean_distance": sum(distances) / len(distances),
        "distance_distribution": dict(sorted(distance_distribution.items())),
    }


def _human_review_report(results: list[ExampleResult]) -> dict:
    correct = sum(1 for r in results if r.actual_needs_human_review == r.expected_needs_human_review)
    return {"accuracy": correct / len(results)}


def _latency_report(results: list[ExampleResult]) -> dict:
    latencies = [r.latency_ms for r in results]
    return {
        "mean_latency_ms": sum(latencies) / len(latencies),
        # Rate-limit waits inflate single examples a lot, so the median is
        # the latency figure to compare between runs.
        "median_latency_ms": statistics.median(latencies),
        "max_latency_ms": max(latencies),
        "total_retries": sum(r.retry_count for r in results),
        "fallbacks": sum(1 for r in results if r.fell_back),
    }


def _metrics(results: list[ExampleResult]) -> dict:
    """All report numbers in one dict, as stored in eval_runs."""
    priority = _priority_report(results)
    # BSON document keys must be strings; the distance buckets are ints.
    priority["distance_distribution"] = {str(k): v for k, v in priority["distance_distribution"].items()}
    return {
        "category": _category_report(results),
        "priority": priority,
        "needs_human_review": _human_review_report(results),
        "latency": _latency_report(results),
    }


def _print_report(mode: str, results: list[ExampleResult]) -> None:
    cat = _category_report(results)
    pri = _priority_report(results)
    hr = _human_review_report(results)
    lat = _latency_report(results)

    print(f"\n=== {mode.upper()} MODE - {len(results)} examples ===")

    print(f"\nCategory accuracy: {cat['accuracy']:.1%}")
    print(f"{'category':<26}{'precision':>10}{'recall':>9}{'support':>9}")
    for name, m in cat["per_category"].items():
        precision = f"{m['precision']:.0%}" if m["precision"] is not None else "n/a"
        recall = f"{m['recall']:.0%}" if m["recall"] is not None else "n/a"
        print(f"{name:<26}{precision:>10}{recall:>9}{m['support']:>9}")

    print(f"\nPriority exact-match accuracy: {pri['exact_match_accuracy']:.1%}")
    print(f"Priority mean distance (0 = exact, higher = further off): {pri['mean_distance']:.2f}")
    print(f"Priority distance distribution {{levels off: count}}: {pri['distance_distribution']}")

    print(f"\nneeds_human_review accuracy: {hr['accuracy']:.1%}")

    print(
        f"\nLatency - mean: {lat['mean_latency_ms']:.0f}ms, median: {lat['median_latency_ms']:.0f}ms, "
        f"max: {lat['max_latency_ms']}ms | total retries across all examples: {lat['total_retries']}"
        f" | fallbacks: {lat['fallbacks']}"
    )

    misses = [r for r in results if r.actual_category != r.expected_category]
    if misses:
        print(f"\nCategory misses ({len(misses)}):")
        for r in misses:
            print(
                f"  {r.service}/{r.environment}: expected {r.expected_category}, "
                f"got {r.actual_category} (confidence={r.confidence:.2f}) - {r.notes}"
            )


def _print_comparison(single_results: list[ExampleResult], decomposed_results: list[ExampleResult]) -> None:
    print("\n=== COMPARISON: single-call vs decomposed ===")

    disagreements = [
        (s, d)
        for s, d in zip(single_results, decomposed_results)
        if s.actual_category != d.actual_category or s.actual_priority != d.actual_priority
    ]
    print(f"Disagreements between the two modes: {len(disagreements)}/{len(single_results)}")
    for s, d in disagreements:
        print(f"  {s.service}/{s.environment} (expected {s.expected_category}/{s.expected_priority}):")
        print(f"    single:     {s.actual_category}/{s.actual_priority}")
        print(f"    decomposed: {d.actual_category}/{d.actual_priority}")

    def _acc(results: list[ExampleResult], field: str) -> float:
        return sum(1 for r in results if getattr(r, f"actual_{field}") == getattr(r, f"expected_{field}")) / len(
            results
        )

    def _mean_latency(results: list[ExampleResult]) -> float:
        return sum(r.latency_ms for r in results) / len(results)

    print(f"\n{'metric':<28}{'single':>10}{'decomposed':>13}")
    print(f"{'category accuracy':<28}{_acc(single_results, 'category'):>10.1%}{_acc(decomposed_results, 'category'):>13.1%}")
    print(f"{'priority exact-match':<28}{_acc(single_results, 'priority'):>10.1%}{_acc(decomposed_results, 'priority'):>13.1%}")
    print(f"{'mean latency (ms)':<28}{_mean_latency(single_results):>10.0f}{_mean_latency(decomposed_results):>13.0f}")


async def _save(batch_id: ObjectId, mode: str, examples_total: int, limit: int | None,
                results: list[ExampleResult], traces: list[LLMRun]) -> None:
    eval_run_id = await save_eval_run(batch_id=batch_id,
                                      mode=mode,
                                      model=llm_client.GROQ_MODEL,
                                      dataset_path=_DATA_PATH,
                                      examples_total=examples_total,
                                      limit=limit,
                                      metrics=_metrics(results),
                                      results=results,
                                      traces=traces)
    print(f"Saved {mode} run to MongoDB: eval_runs {eval_run_id}, {len(traces)} traces in llm_runs")


def _fmt_spread(row: dict, field: str) -> str:
    """'95.1% ±1.7 (93.9-97.6)' - stddev is missing until there are two runs."""
    mean, stddev, low, high = (row[f"{field}_{k}"] for k in ("mean", "stddev", "min", "max"))
    spread = f" ±{stddev:.1%}" if stddev is not None else ""
    return f"{mean:.1%}{spread} ({low:.1%}-{high:.1%})"


async def _print_history() -> None:
    recent, summary = await load_history(_DATA_PATH)
    print("=== Recent eval runs (newest first) ===")
    print(f"{'when (UTC)':<18}{'mode':<12}{'examples':>9}{'category':>10}{'priority':>10}{'median ms':>11}"
          f"{'fallbacks':>11}  dataset")
    for run in recent:
        m = run["metrics"]
        examples = run["dataset"]["examples"] if run["limit"] is None else f"{run['limit']}*"
        print(f"{run['created_at']:%Y-%m-%d %H:%M}  {run['mode']:<12}{examples:>9}"
              f"{m['category']['accuracy']:>10.1%}{m['priority']['exact_match_accuracy']:>10.1%}"
              f"{m['latency']['median_latency_ms']:>11.0f}{m['latency']['fallbacks']:>11}  {run['dataset']['sha256']}")
    print("(* = --limit run, not comparable)")

    print(f"\n=== Complete runs on the current dataset ({dataset_fingerprint(_DATA_PATH)}), per mode ===")
    for row in summary:
        print(f"{row['_id']['mode']} / {row['_id']['model']} - {row['runs']} run(s), last {row['last_run']:%Y-%m-%d %H:%M}")
        print(f"  category accuracy:    {_fmt_spread(row, 'category')}")
        print(f"  priority exact match: {_fmt_spread(row, 'priority')}  mean distance {row['mean_distance']:.2f}")
        print(f"  median latency:       {row['median_latency_ms']:.0f}ms (mean over runs)")
    if not summary:
        print("(none yet - run with --save and no --limit)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["single", "decomposed", "both"],
        default="decomposed" if llm_client.TASK_DECOMPOSITION else "single",
        help="Which pipeline to evaluate (default: whatever TASK_DECOMPOSITION is currently set to)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples (smoke test)")
    parser.add_argument("--save", action="store_true",
                        help="Also store results (eval_runs) and traces (llm_runs) in MongoDB")
    parser.add_argument("--history", action="store_true",
                        help="Print past runs from MongoDB instead of running the eval")
    args = parser.parse_args()

    try:
        # One event loop for the whole run: the Groq client's connection pool
        # belongs to the loop it was first used in.
        asyncio.run(_run(args))
    except PyMongoError as e:
        # The eval itself may have finished and been printed above; only
        # storing it failed. Fail loudly - --save was asked for explicitly.
        sys.exit(f"MongoDB error: {e}")


async def _run(args: argparse.Namespace) -> None:
    if args.history:
        await _print_history()
        return

    examples = _load_examples(args.limit)
    examples_total = len(_load_examples(None))
    print(f"Loaded {len(examples)} examples from {_DATA_PATH}")
    modes = ["single", "decomposed"] if args.mode == "both" else [args.mode]
    # One id for the whole invocation, so the two modes of a --mode both
    # run can be paired up later.
    batch_id = ObjectId()

    try:
        runs = {mode: await _run_pipeline(examples, mode) for mode in modes}
    finally:
        await llm_client.aclose_client()
    for mode, (results, _) in runs.items():
        _print_report(mode, results)
    if args.mode == "both":
        _print_comparison(runs["single"][0], runs["decomposed"][0])
    if args.save:
        print()
        for mode, (results, traces) in runs.items():
            await _save(batch_id, mode, examples_total, args.limit, results, traces)


if __name__ == "__main__":
    main()
