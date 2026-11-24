from datetime import timezone

from app.llm_trace import AttemptTrace, LLMRun, StepTrace, prompt_fingerprint


def step(name: str, statuses: list[str], latency_ms: int, fell_back: bool = False) -> StepTrace:
    return StepTrace(name=name, system_prompt_fingerprint="abc", user_prompt="u", latency_ms=latency_ms,
                     fell_back=fell_back,
                     attempts=[AttemptTrace(n=i + 1, status=s, latency_ms=10) for i, s in enumerate(statuses)])


def test_step_retry_count_is_attempts_minus_one():
    assert step("classify", ["invalid_json", "ok"], 20).retry_count == 1
    assert step("classify", [], 0).retry_count == 0


def test_run_totals_add_up_the_steps():
    run = LLMRun(mode="decomposed", model="m", steps=[
        step("classify", ["timeout", "ok"], 1200),
        step("prioritize", ["ok"], 300),
    ])
    assert run.latency_ms == 1500
    assert run.retry_count == 1
    assert not run.fell_back
    assert LLMRun(mode="single", model="m", steps=[step("analyze", ["timeout"] * 3, 9000, fell_back=True)]).fell_back


def test_dump_includes_computed_totals_and_a_utc_datetime():
    dumped = LLMRun(mode="single", model="m", steps=[step("analyze", ["ok"], 700)]).model_dump(mode="python")
    # Stored in the Mongo document, so they can be filtered and aggregated on.
    assert dumped["latency_ms"] == 700 and dumped["retry_count"] == 0 and dumped["fell_back"] is False
    assert dumped["steps"][0]["retry_count"] == 0
    # A datetime (BSON date, which the TTL index needs), not an ISO string.
    assert dumped["created_at"].tzinfo is timezone.utc


def test_prompt_fingerprint_is_short_stable_and_changes_with_the_prompt():
    fp = prompt_fingerprint("You are an SRE assistant.")
    assert len(fp) == 12
    assert fp == prompt_fingerprint("You are an SRE assistant.")
    assert fp != prompt_fingerprint("You are an SRE assistant!")
