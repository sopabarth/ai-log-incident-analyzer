"""
Groq integration - turns a normalized log entry into an IncidentAnalysis.

Task decomposition: this used to be one LLM call asking for everything at
once (category, root cause, priority, confidence, review flag). It's now
two focused calls:

  1. classify  - category, root_cause_summary, confidence, needs_human_review
  2. prioritize - priority, priority_reasoning, given the classification
                  from step 1 as additional context

Why priority still needs its own LLM call rather than a deterministic
(category, environment) -> priority lookup table: checking that idea
against data/synthetic_logs.py's own ground truth shows the same category
in the same environment legitimately spans multiple priorities (e.g.
auth_failure in prod is labeled critical, high, AND low across different
examples) - the deciding factor is the blast radius described in the
error text itself ("all requests rejected" vs "one user locked out"),
which only reading the actual text can capture. A lookup table keyed on
category+environment alone would regress accuracy, not just simplify code.

TASK_DECOMPOSITION env var (default false) toggles between a single
combined call (_analyze_single_call, the way this module worked before
decomposition) and the two-call pipeline above.
Kept side by side on purpose - it's a direct, switchable comparison between
"one big prompt" and "decomposed pipeline" against the exact same retry/
fallback machinery, which is a more convincing demonstration than deleting
the old approach outright.

Retry/fallback policy, shared by all three steps via _call_with_retry
(which takes the plain prompt/model/fallback values it needs directly,
rather than a callable to invoke - see its docstring for why):
  - Malformed model output (invalid JSON, or JSON that fails schema
    validation) is retried immediately - it's a one-off bad generation, not
    a timing issue, so there's no reason to wait before asking again.
  - Transient Groq API errors (connection drop, timeout, rate limit, 5xx)
    are retried with a short backoff, since hammering an already-
    struggling/throttled API immediately tends to make things worse.
  - Both of the above, if still failing after LLM_MAX_ATTEMPTS attempts,
    fall back to a generic result for that step instead of failing the
    request outright - a burst of bad model output shouldn't take the
    whole endpoint down.
  - Non-retryable errors (missing/invalid API key, malformed request, etc.)
    are NOT retried and NOT papered over with a fallback - retrying a
    guaranteed-to-fail auth error just burns time, and silently returning
    "unknown incident" for a broken deployment would hide an ops problem
    that needs to be loud, not classified.
"""

import json
import os
import time
from typing import TypeVar

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel, ValidationError

from app.schemas import ClassificationResult, ErrorCategory, IncidentAnalysis, Priority, PriorityResult

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", 3))
RETRY_BACKOFF_SECONDS = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", 0.5))
TASK_DECOMPOSITION = os.getenv("TASK_DECOMPOSITION", "false").strip().lower() in ("true", "1", "yes")
_RETRYABLE_API_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

_client: Groq | None = None
_T = TypeVar("_T", bound=BaseModel)


class LLMResponseError(Exception):
    """Raised when a single attempt's response can't be parsed into the expected model."""


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set - copy .env.example to .env and fill it in."
            )
        _client = Groq(api_key=api_key)
    return _client


def analyze_incident(service: str, environment: str, normalized_text: str) -> tuple[IncidentAnalysis, int, int]:
    """Dispatches to the decomposed or single-call pipeline per TASK_DECOMPOSITION."""
    if TASK_DECOMPOSITION:
        return _analyze_decomposed(service, environment, normalized_text)
    return _analyze_single_call(service, environment, normalized_text)


def _run_completion(system_prompt: str, user_prompt: str, response_model: type[_T]) -> _T:
    """One unretried Groq round-trip, parsed into `response_model`. Raises
    LLMResponseError on unusable output, or lets Groq API exceptions
    (network/auth/rate-limit/...) propagate as-is."""
    client = _get_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"Model did not return valid JSON: {e}\nRaw: {content!r}") from e

    try:
        return response_model(**data)
    except ValidationError as e:
        raise LLMResponseError(f"Model JSON failed schema validation: {e}\nRaw: {data!r}") from e


def _call_with_retry(step_name: str,
                     system_prompt: str,
                     user_prompt: str,
                     response_model: type[_T],
                     fallback: _T, ) -> tuple[_T, int, int]:
    """
    Shared retry loop for one pipeline step. Takes the plain ingredients
    for the Groq call (prompts + expected response type) and calls
    _run_completion() itself.

    Returns (result, latency_ms, retry_count); returns `fallback` instead
    if every attempt fails on a retryable error. See module docstring for
    the full retry/fallback policy.
    """
    started = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            result = _run_completion(system_prompt, user_prompt, response_model)
            latency_ms = int((time.monotonic() - started) * 1000)
            return result, latency_ms, attempt - 1
        except LLMResponseError as e:
            # Bad generation, not a timing issue - retry right away.
            last_error = e
            print(f"[llm_client:{step_name}] attempt {attempt}/{MAX_ATTEMPTS} failed (malformed output): {e}")
        except _RETRYABLE_API_ERRORS as e:
            # Transient infra issue - back off a bit before hammering again.
            last_error = e
            print(f"[llm_client:{step_name}] attempt {attempt}/{MAX_ATTEMPTS} failed ({type(e).__name__}): {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    # Exhausted every attempt on retryable failures only - fall back instead
    # of failing the request outright. print() is a placeholder here; proper
    # structured logging is its own later step.
    print(f"[llm_client:{step_name}] all {MAX_ATTEMPTS} attempts failed, falling back. Last error: {last_error}")
    latency_ms = int((time.monotonic() - started) * 1000)
    return fallback, latency_ms, MAX_ATTEMPTS - 1


# --- Single-call mode (TASK_DECOMPOSITION=false) ---
#
# Same job as classify + prioritize combined into one prompt/one call - kept
# for direct comparison against the decomposed pipeline above, reusing the
# same _call_with_retry helper.

_SINGLE_CALL_SYSTEM_PROMPT = f"""You are an SRE assistant that triages application error logs.

Given a normalized error/stack trace plus its service context, respond with a
JSON object with EXACTLY these fields and nothing else:

- "category": one of {[c.value for c in ErrorCategory]}
- "root_cause_summary": 1-2 plain-language sentences, under 300 characters.
- "priority": one of {[p.value for p in Priority]}
- "priority_reasoning": one short sentence, under 200 characters.
- "confidence": float between 0.0 and 1.0, how confident you are in this
  classification given the evidence.
- "needs_human_review": boolean, true if the trace is ambiguous, truncated,
  contradictory, or doesn't clearly match a known pattern - in that case
  still give your best-guess category/priority, but set this true and lower
  your confidence accordingly. Prefer true over guessing wildly.

Judge priority using the service, environment, and blast radius described in
the context - the same error type can be "critical" in prod on a checkout
path and "low" in a dev environment.

Return ONLY the JSON object, no markdown fences, no commentary.
"""


def _analyze_single_call(service: str, environment: str, normalized_text: str) -> tuple[IncidentAnalysis, int, int]:
    # Same user prompt shape as _classify - both just need service/environment/error.
    user_prompt = _build_classify_user_prompt(service, environment, normalized_text)
    return _call_with_retry("analyze",
                            _SINGLE_CALL_SYSTEM_PROMPT,
                            user_prompt,
                            IncidentAnalysis,
                            _fallback_single_call_analysis())


def _fallback_single_call_analysis() -> IncidentAnalysis:
    return IncidentAnalysis(category=ErrorCategory.UNKNOWN,
                            root_cause_summary=("Automated analysis failed after repeated attempts - the model "
                                                "did not return a usable classification for this error."),
                            priority=Priority.MEDIUM,
                            priority_reasoning="Priority could not be determined automatically; needs manual triage.",
                            confidence=0.0,
                            needs_human_review=True)


# --- Task decomposition mode (TASK_DECOMPOSITION=true) ---

def _analyze_decomposed(service: str, environment: str, normalized_text: str) -> tuple[IncidentAnalysis, int, int]:
    """
    Runs the classify -> prioritize pipeline (each step independently
    retried/falls back - see module docstring) and merges the results into
    the IncidentAnalysis shape the rest of the app already expects.

    Returns (analysis, total_latency_ms, total_retry_count) - latency and
    retry count are summed across both steps.
    """
    classification, classify_latency_ms, classify_retries = _classify(service, environment, normalized_text)
    priority_result, prioritize_latency_ms, prioritize_retries = _prioritize(service,
                                                                             environment,
                                                                             normalized_text,
                                                                             classification.category)

    analysis = IncidentAnalysis(category=classification.category,
                                root_cause_summary=classification.root_cause_summary,
                                priority=priority_result.priority,
                                priority_reasoning=priority_result.priority_reasoning,
                                confidence=classification.confidence,
                                needs_human_review=classification.needs_human_review)
    total_latency_ms = classify_latency_ms + prioritize_latency_ms
    total_retry_count = classify_retries + prioritize_retries
    return analysis, total_latency_ms, total_retry_count


# --- Step 1: classification ---

_CLASSIFY_SYSTEM_PROMPT = f"""You are an SRE assistant that classifies application error logs.

Given a normalized error/stack trace plus its service context, respond with a
JSON object with EXACTLY these fields and nothing else:

- "category": one of {[c.value for c in ErrorCategory]}
- "root_cause_summary": 1-2 plain-language sentences, under 300 characters.
- "confidence": float between 0.0 and 1.0, how confident you are in this
  classification given the evidence.
- "needs_human_review": boolean, true if the trace is ambiguous, truncated,
  contradictory, or doesn't clearly match a known pattern - in that case
  still give your best-guess category, but set this true and lower your
  confidence accordingly. Prefer true over guessing wildly.

Return ONLY the JSON object, no markdown fences, no commentary.
"""


def _build_classify_user_prompt(service: str, environment: str, normalized_text: str) -> str:
    return (f"service: {service}\n"
            f"environment: {environment}\n"
            f"error:\n{normalized_text}")


def _classify(service: str, environment: str, normalized_text: str) -> tuple[ClassificationResult, int, int]:
    user_prompt = _build_classify_user_prompt(service, environment, normalized_text)
    return _call_with_retry("classify",
                            _CLASSIFY_SYSTEM_PROMPT,
                            user_prompt,
                            ClassificationResult,
                            _fallback_classification())


def _fallback_classification() -> ClassificationResult:
    return ClassificationResult(category=ErrorCategory.UNKNOWN,
                                root_cause_summary=("Automated classification failed after repeated attempts - the "
                                                    "model did not return a usable result for this error."),
                                confidence=0.0,
                                needs_human_review=True)


# --- Step 2: prioritization ---

_PRIORITIZE_SYSTEM_PROMPT = f"""You are an SRE assistant that sets the *priority* of an
already-classified application incident.

You are given the error category already determined for this incident, its
service/environment context, and the normalized error text. Respond with a
JSON object with EXACTLY these fields and nothing else:

- "priority": one of {[p.value for p in Priority]}
- "priority_reasoning": one short sentence, under 200 characters.

Judge priority using the blast radius actually described in the error text
(how many requests/users are affected, whether the failure is total or
partial, whether there's a working retry/fallback) together with the
service and environment - the same error category can be "critical" in
prod when it affects all requests on a checkout path, and "low" for a
single affected user or a background job with no user impact.

Return ONLY the JSON object, no markdown fences, no commentary.
"""


def _build_prioritize_user_prompt(service: str, environment: str, normalized_text: str, category: ErrorCategory) -> str:
    return (f"category: {category.value}\n"
            f"service: {service}\n"
            f"environment: {environment}\n"
            f"error:\n{normalized_text}")


def _prioritize(service: str,
                environment: str,
                normalized_text: str,
                category: ErrorCategory) -> tuple[PriorityResult, int, int]:
    user_prompt = _build_prioritize_user_prompt(service, environment, normalized_text, category)
    return _call_with_retry("prioritize",
                            _PRIORITIZE_SYSTEM_PROMPT,
                            user_prompt,
                            PriorityResult,
                            _fallback_priority())


def _fallback_priority() -> PriorityResult:
    # MEDIUM is a deliberate middle ground here, same reasoning as the
    # single-call fallback: LOW risks a genuinely serious incident being
    # ignored, CRITICAL risks paging someone over what might be nothing.
    return PriorityResult(priority=Priority.MEDIUM,
                          priority_reasoning="Priority could not be determined automatically; needs manual triage.")
