"""The eval's scoring, on hand-built results with hand-computed expected values."""

import pytest

from eval.run_eval import ExampleResult, _category_report, _fmt_spread, _latency_report, _metrics, _priority_report


def result(expected_category="database_timeout", actual_category="database_timeout", expected_priority="high",
           actual_priority="high", expected_review=False, actual_review=False, latency_ms=1000, retry_count=0,
           fell_back=False) -> ExampleResult:
    return ExampleResult(service="svc", environment="prod", expected_category=expected_category,
                         actual_category=actual_category, expected_priority=expected_priority,
                         actual_priority=actual_priority, expected_needs_human_review=expected_review,
                         actual_needs_human_review=actual_review, confidence=0.9, latency_ms=latency_ms,
                         retry_count=retry_count, fell_back=fell_back, error_hash="h", notes="")


# Five results, worked out by hand:
#   category: 4/5 right (the third is auth_failure called database_timeout)
#   priority: distances 0, 1, 0, 3, 0  -> exact 3/5, mean 0.8
#   review:   wrong only in the fourth  -> 4/5
#   latency:  1, 2, 3, 4, 10 s          -> mean 4 s, median 3 s, max 10 s
RESULTS = [
    result(latency_ms=1000),
    result(actual_priority="medium", latency_ms=2000, retry_count=1),
    result(expected_category="auth_failure", expected_priority="low", actual_priority="low", latency_ms=3000),
    result(expected_category="auth_failure", actual_category="auth_failure", expected_priority="critical",
           actual_priority="low", actual_review=True, latency_ms=4000, retry_count=2),
    result(expected_category="unknown", actual_category="unknown", latency_ms=10000, retry_count=2, fell_back=True),
]


def test_category_report():
    report = _category_report(RESULTS)
    assert report["accuracy"] == pytest.approx(0.8)
    per = report["per_category"]
    # database_timeout: called 3 times, right twice; labeled twice, found both.
    assert per["database_timeout"] == {"precision": pytest.approx(2 / 3), "recall": 1.0, "support": 2}
    # auth_failure: called once (right); labeled twice, found one.
    assert per["auth_failure"] == {"precision": 1.0, "recall": 0.5, "support": 2}


def test_category_report_leaves_undefined_ratios_as_none():
    per = _category_report([result(expected_category="auth_failure")])["per_category"]
    assert per["auth_failure"]["precision"] is None  # never predicted
    assert per["database_timeout"]["recall"] is None  # never the label


def test_priority_report():
    report = _priority_report(RESULTS)
    assert report["exact_match_accuracy"] == pytest.approx(0.6)
    assert report["mean_distance"] == pytest.approx(0.8)
    assert report["distance_distribution"] == {0: 3, 1: 1, 3: 1}


def test_latency_report_uses_the_median_and_counts_fallbacks():
    report = _latency_report(RESULTS)
    assert report["mean_latency_ms"] == 4000
    assert report["median_latency_ms"] == 3000
    assert report["max_latency_ms"] == 10000
    assert report["total_retries"] == 5
    assert report["fallbacks"] == 1


def test_metrics_are_ready_for_bson():
    metrics = _metrics(RESULTS)
    assert set(metrics) == {"category", "priority", "needs_human_review", "latency"}
    assert metrics["needs_human_review"]["accuracy"] == pytest.approx(0.8)
    # BSON keys must be strings; the report's own distance buckets are ints.
    assert metrics["priority"]["distance_distribution"] == {"0": 3, "1": 1, "3": 1}


def test_fmt_spread_shows_stddev_only_once_there_are_two_runs():
    one_run = {"category_mean": 0.976, "category_stddev": None, "category_min": 0.976, "category_max": 0.976}
    assert _fmt_spread(one_run, "category") == "97.6% (97.6%-97.6%)"
    two_runs = {"category_mean": 0.9635, "category_stddev": 0.0175, "category_min": 0.951, "category_max": 0.976}
    assert _fmt_spread(two_runs, "category") == "96.4% ±1.8% (95.1%-97.6%)"
