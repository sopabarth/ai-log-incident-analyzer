"""
llm_client without Groq: _run_completion (the one function that talks to
the API) is replaced by a scripted fake, and time.sleep by a recorder, so
the retry policy can be checked exactly and instantly.
"""

import json

import httpx
import pytest
from groq import APIConnectionError, APITimeoutError, AuthenticationError, InternalServerError, RateLimitError

import app.llm_client as llm
from app.llm_trace import prompt_fingerprint
from app.schemas import ClassificationResult, ErrorCategory, IncidentAnalysis, Priority

_REQUEST = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(cls, status: int):
    return cls(f"HTTP {status}", response=httpx.Response(status, request=_REQUEST), body=None)


def rate_limited():
    return _status_error(RateLimitError, 429)


def server_error():
    return _status_error(InternalServerError, 500)


ANALYSIS = json.dumps({"category": "database_timeout", "root_cause_summary": "Pool exhausted.", "priority": "high",
                       "priority_reasoning": "Checkout affected.", "confidence": 0.9, "needs_human_review": False})
CLASSIFICATION = json.dumps({"category": "auth_failure", "root_cause_summary": "Token expired.",
                             "confidence": 0.8, "needs_human_review": False})
PRIORITY = json.dumps({"priority": "low", "priority_reasoning": "One user affected."})


class FakeModel:
    """Plays back scripted replies: a string is returned as the model's text,
    an exception is raised as if the API call failed."""

    def __init__(self):
        self.replies: list = []
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture
def model(monkeypatch) -> FakeModel:
    fake = FakeModel()
    monkeypatch.setattr(llm, "_run_completion", fake)
    # Pin the settings .env may override, so tests don't depend on it.
    monkeypatch.setattr(llm, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(llm, "RETRY_BACKOFF_SECONDS", 0.5)
    monkeypatch.setattr(llm, "TASK_DECOMPOSITION", False)
    monkeypatch.setattr(llm, "GROQ_MODEL", "test-model")
    return fake


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(llm.time, "sleep", recorded.append)
    return recorded


# --- _parse_response ---

def test_parse_response_returns_the_model():
    assert llm._parse_response(CLASSIFICATION, ClassificationResult).category is ErrorCategory.AUTH_FAILURE


@pytest.mark.parametrize("content, status, message_part", [
    ("not json", "invalid_json", "invalid JSON"),
    ("[1, 2]", "validation_error", "expected a JSON object, got list"),
    ('{"category": "made_up"}', "validation_error", "category"),
    (json.dumps({**json.loads(CLASSIFICATION), "confidence": 2}), "validation_error", "confidence"),
])
def test_parse_response_says_what_was_wrong(content, status, message_part):
    with pytest.raises(llm.LLMResponseError) as exc:
        llm._parse_response(content, ClassificationResult)
    assert exc.value.status == status
    assert message_part in str(exc.value)
    # The raw text lives in the trace, not repeated in the message.
    assert content not in str(exc.value)


@pytest.mark.parametrize("error, status", [
    (rate_limited(), "rate_limited"),
    (APITimeoutError(request=_REQUEST), "timeout"),  # subclass of APIConnectionError
    (APIConnectionError(request=_REQUEST), "connection_error"),
    (server_error(), "server_error"),
])
def test_api_errors_map_to_attempt_statuses(error, status):
    assert llm._api_error_status(error) == status


# --- _call_with_retry ---

def call(fallback=None):
    fallback = fallback or llm._fallback_classification()
    return llm._call_with_retry("classify", "system prompt", "user prompt", ClassificationResult, fallback)


def test_first_attempt_succeeds(model, sleeps):
    model.replies = [CLASSIFICATION]
    result, step = call()

    assert result.category is ErrorCategory.AUTH_FAILURE
    assert [a.status for a in step.attempts] == ["ok"]
    assert step.attempts[0].raw_response == CLASSIFICATION
    assert step.result == result.model_dump(mode="json")
    assert step.retry_count == 0 and not step.fell_back
    assert step.name == "classify"
    assert step.user_prompt == "user prompt"
    assert step.system_prompt_fingerprint == prompt_fingerprint("system prompt")
    assert sleeps == []


def test_malformed_output_is_retried_immediately(model, sleeps):
    model.replies = ["not json", '{"category": "made_up"}', CLASSIFICATION]
    result, step = call()

    assert result.category is ErrorCategory.AUTH_FAILURE
    assert [a.status for a in step.attempts] == ["invalid_json", "validation_error", "ok"]
    # Every response that came back is kept, failed ones included.
    assert [a.raw_response for a in step.attempts] == ["not json", '{"category": "made_up"}', CLASSIFICATION]
    assert [a.n for a in step.attempts] == [1, 2, 3]
    assert step.retry_count == 2
    assert sleeps == []  # a bad generation is not a timing problem


def test_transient_api_errors_back_off(model, sleeps):
    model.replies = [rate_limited(), APITimeoutError(request=_REQUEST), CLASSIFICATION]
    _, step = call()

    assert [a.status for a in step.attempts] == ["rate_limited", "timeout", "ok"]
    assert step.attempts[0].raw_response is None  # no response came back
    assert step.attempts[0].error.startswith("RateLimitError")
    assert sleeps == [0.5, 1.0]  # backoff x attempt


def test_exhausted_attempts_fall_back(model, sleeps):
    model.replies = [server_error(), "not json", APIConnectionError(request=_REQUEST)]
    fallback = llm._fallback_classification()
    result, step = call(fallback)

    assert result == fallback
    assert step.fell_back
    assert step.result == fallback.model_dump(mode="json")
    assert [a.status for a in step.attempts] == ["server_error", "invalid_json", "connection_error"]
    assert step.retry_count == 2
    assert sleeps == [0.5]  # after the API error on attempt 1, not after the last attempt


def test_non_retryable_errors_propagate(model, sleeps):
    model.replies = [_status_error(AuthenticationError, 401), CLASSIFICATION]
    with pytest.raises(AuthenticationError):
        call()
    assert len(model.calls) == 1


def test_max_attempts_setting_is_honoured(model, sleeps, monkeypatch):
    monkeypatch.setattr(llm, "MAX_ATTEMPTS", 1)
    model.replies = ["not json"]
    _, step = call()
    assert step.fell_back and len(step.attempts) == 1


# --- analyze_incident ---

def test_single_call_mode(model, sleeps):
    model.replies = ["not json", ANALYSIS]
    analysis, run = llm.analyze_incident("checkout-api", "prod", "TimeoutError: pool")

    assert isinstance(analysis, IncidentAnalysis) and analysis.priority is Priority.HIGH
    assert run.mode == "single" and run.model == "test-model"
    assert [s.name for s in run.steps] == ["analyze"]
    assert run.retry_count == 1 and not run.fell_back
    system_prompt, user_prompt = model.calls[0]
    assert user_prompt == "service: checkout-api\nenvironment: prod\nerror:\nTimeoutError: pool"
    assert run.steps[0].system_prompt_fingerprint == prompt_fingerprint(system_prompt)


def test_decomposed_mode_runs_classify_then_prioritize(model, sleeps, monkeypatch):
    monkeypatch.setattr(llm, "TASK_DECOMPOSITION", True)
    model.replies = [CLASSIFICATION, "not json", PRIORITY]
    analysis, run = llm.analyze_incident("auth-api", "staging", "401 Unauthorized")

    assert run.mode == "decomposed"
    assert [s.name for s in run.steps] == ["classify", "prioritize"]
    # The merged analysis takes each field from the step that produced it.
    assert analysis.category is ErrorCategory.AUTH_FAILURE and analysis.confidence == 0.8
    assert analysis.priority is Priority.LOW
    # Totals are summed across the steps.
    assert run.retry_count == 1
    assert run.latency_ms == sum(s.latency_ms for s in run.steps)
    # Prioritize gets the category classify decided, and its own system prompt.
    assert model.calls[1][1].startswith("category: auth_failure\n")
    assert run.steps[0].system_prompt_fingerprint != run.steps[1].system_prompt_fingerprint


def test_decomposed_fallback_in_one_step_marks_the_run(model, sleeps, monkeypatch):
    monkeypatch.setattr(llm, "TASK_DECOMPOSITION", True)
    model.replies = [CLASSIFICATION, "bad", "bad", "bad"]
    analysis, run = llm.analyze_incident("auth-api", "prod", "401")

    assert run.fell_back
    assert [s.fell_back for s in run.steps] == [False, True]
    # Classification survives; only priority is the fallback's.
    assert analysis.category is ErrorCategory.AUTH_FAILURE
    assert analysis.priority is Priority.MEDIUM
