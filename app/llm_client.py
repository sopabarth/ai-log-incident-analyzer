"""
Groq integration - turns a normalized log entry into an IncidentAnalysis.

Retry/fallback policy:
  - Malformed model output (invalid JSON, or JSON that fails IncidentAnalysis
    validation) is retried immediately - it's a one-off bad generation, not a
    timing issue, so there's no reason to wait before asking again.
  - Transient Groq API errors (connection drop, timeout, rate limit, 5xx) are
    retried with a short backoff, since hammering an already-struggling/
    throttled API immediately tends to make things worse.
  - Both of the above, if still failing after LLM_MAX_ATTEMPTS attempts, fall
    back to a generic "uncategorized, needs manual review" IncidentAnalysis
    instead of failing the request outright - a burst of bad model output
    shouldn't take the whole endpoint down.
  - Non-retryable errors (missing/invalid API key, malformed request, etc.)
    are NOT retried and NOT papered over with a fallback - retrying a
    guaranteed-to-fail auth error just burns time, and silently returning
    "unknown incident" for a broken deployment would hide an ops problem
    that needs to be loud, not classified.
"""

import json
import os
import time

from dotenv import load_dotenv
from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)
from pydantic import ValidationError

from app.schemas import ErrorCategory, IncidentAnalysis, Priority

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", 3))
RETRY_BACKOFF_SECONDS = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", 0.5))

# Errors worth retrying: transient/infrastructure issues where the same
# request plausibly succeeds a moment later. Anything else (bad API key,
# malformed request, permission errors, ...) is left to propagate.
_RETRYABLE_API_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

_client: Groq | None = None


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


class LLMResponseError(Exception):
    """Raised when a single attempt's response can't be parsed into IncidentAnalysis."""


_SYSTEM_PROMPT = f"""You are an SRE assistant that triages application error logs.

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


def _build_user_prompt(service: str, environment: str, normalized_text: str) -> str:
    return (
        f"service: {service}\n"
        f"environment: {environment}\n"
        f"error:\n{normalized_text}"
    )


def _call_once(service: str, environment: str, normalized_text: str) -> IncidentAnalysis:
    """One unretried Groq round-trip. Raises LLMResponseError on unusable output,
    or lets Groq API exceptions (network/auth/rate-limit/...) propagate as-is."""
    client = _get_client()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(service, environment, normalized_text)},
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
        return IncidentAnalysis(**data)
    except ValidationError as e:
        raise LLMResponseError(f"Model JSON failed schema validation: {e}\nRaw: {data!r}") from e


def _fallback_analysis() -> IncidentAnalysis:
    """
    Used when every attempt failed. Deliberately not a guess: confidence=0.0
    and needs_human_review=True make it obvious this wasn't a real
    classification, and MEDIUM keeps it from being silently ignored (LOW)
    or paging someone for what might be nothing (CRITICAL) purely because
    the model had a bad run.
    """
    return IncidentAnalysis(
        category=ErrorCategory.UNKNOWN,
        root_cause_summary=(
            "Automated analysis failed after repeated attempts - the model "
            "did not return a usable classification for this error."
        ),
        priority=Priority.MEDIUM,
        priority_reasoning="Priority could not be determined automatically; needs manual triage.",
        confidence=0.0,
        needs_human_review=True,
    )


def analyze_incident(
    service: str, environment: str, normalized_text: str
) -> tuple[IncidentAnalysis, int, int]:
    """
    Calls Groq with retry on malformed output and transient API errors, up to
    MAX_ATTEMPTS attempts total, then falls back to _fallback_analysis() if
    none succeeded. Non-retryable errors (bad API key, malformed request,
    ...) propagate immediately without retrying or falling back.

    Returns (analysis, total_latency_ms, retry_count) - retry_count is how
    many attempts beyond the first were needed (0 if it succeeded on the
    first try).
    """
    started = time.monotonic()
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            analysis = _call_once(service, environment, normalized_text)
            latency_ms = int((time.monotonic() - started) * 1000)
            return analysis, latency_ms, attempt - 1
        except LLMResponseError as e:
            # Bad generation, not a timing issue - retry right away.
            last_error = e
            print(f"[llm_client] attempt {attempt}/{MAX_ATTEMPTS} failed (malformed output): {e}")
        except _RETRYABLE_API_ERRORS as e:
            # Transient infra issue - back off a bit before hammering again.
            last_error = e
            print(f"[llm_client] attempt {attempt}/{MAX_ATTEMPTS} failed ({type(e).__name__}): {e}")
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        # Anything else (AuthenticationError, BadRequestError, the RuntimeError
        # from a missing API key, ...) is not caught here - it propagates
        # immediately, since retrying it can't succeed and a fallback would
        # hide a broken deployment behind a fake "unknown incident".

    # Exhausted every attempt on retryable failures only - fall back instead
    # of failing the request outright. print() is a placeholder here; proper
    # structured logging is its own later step.
    print(f"[llm_client] all {MAX_ATTEMPTS} attempts failed, falling back. Last error: {last_error}")
    latency_ms = int((time.monotonic() - started) * 1000)
    return _fallback_analysis(), latency_ms, MAX_ATTEMPTS - 1
