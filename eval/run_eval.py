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
    python -m eval.run_eval --save          # also write raw results under eval/results/
"""

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import app.llm_client as llm_client
from app.parser import normalize_raw_text

_DATA_PATH = Path(__file__).parent.parent / "data" / "synthetic_logs.json"
_RESULTS_DIR = Path(__file__).parent / "results"

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
    notes: str


def _load_examples(limit: int | None) -> list[dict]:
    examples = json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    return examples[:limit] if limit else examples


def _run_one(example: dict) -> ExampleResult:
    normalized = normalize_raw_text(example["raw_text"])
    analysis, latency_ms, retry_count = llm_client.analyze_incident(
        example["service"], example["environment"], normalized
    )
    return ExampleResult(
        service=example["service"],
        environment=example["environment"],
        expected_category=example["expected_category"],
        actual_category=analysis.category.value,
        expected_priority=example["expected_priority"],
        actual_priority=analysis.priority.value,
        expected_needs_human_review=example["expected_needs_human_review"],
        actual_needs_human_review=analysis.needs_human_review,
        confidence=analysis.confidence,
        latency_ms=latency_ms,
        retry_count=retry_count,
        notes=example.get("notes", ""),
    )


def _run_pipeline(examples: list[dict], mode: str) -> list[ExampleResult]:
    """
    Runs every example with llm_client.TASK_DECOMPOSITION forced to `mode`
    ("single" or "decomposed"), restoring the original value afterwards.
    Overriding the module attribute directly (rather than the env var)
    works because analyze_incident() reads it fresh on every call, not
    once at import time.
    """
    original = llm_client.TASK_DECOMPOSITION
    llm_client.TASK_DECOMPOSITION = mode == "decomposed"
    try:
        results = []
        for i, example in enumerate(examples, start=1):
            print(
                f"  [{mode}] {i}/{len(examples)}: {example['service']} "
                f"(expected {example['expected_category']})...",
                end=" ",
                flush=True,
            )
            result = _run_one(example)
            verdict = "OK" if result.actual_category == result.expected_category else "MISS"
            print(f"{verdict} -> {result.actual_category}/{result.actual_priority}")
            results.append(result)
        return results
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
        "max_latency_ms": max(latencies),
        "total_retries": sum(r.retry_count for r in results),
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
        f"\nLatency - mean: {lat['mean_latency_ms']:.0f}ms, max: {lat['max_latency_ms']}ms "
        f"| total retries across all examples: {lat['total_retries']}"
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


def _save_results(mode: str, results: list[ExampleResult]) -> None:
    _RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = _RESULTS_DIR / f"{timestamp}_{mode}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
    print(f"Saved raw results to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["single", "decomposed", "both"],
        default="decomposed" if llm_client.TASK_DECOMPOSITION else "single",
        help="Which pipeline to evaluate (default: whatever TASK_DECOMPOSITION is currently set to)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N examples (smoke test)")
    parser.add_argument("--save", action="store_true", help="Also write raw results as JSON under eval/results/")
    args = parser.parse_args()

    examples = _load_examples(args.limit)
    print(f"Loaded {len(examples)} examples from {_DATA_PATH}")

    if args.mode == "both":
        single_results = _run_pipeline(examples, "single")
        decomposed_results = _run_pipeline(examples, "decomposed")
        _print_report("single", single_results)
        _print_report("decomposed", decomposed_results)
        _print_comparison(single_results, decomposed_results)
        if args.save:
            _save_results("single", single_results)
            _save_results("decomposed", decomposed_results)
    else:
        results = _run_pipeline(examples, args.mode)
        _print_report(args.mode, results)
        if args.save:
            _save_results(args.mode, results)


if __name__ == "__main__":
    main()
