"""
Groq integration - turns a normalized log entry into an IncidentAnalysis.

Bare-bones on purpose for now: one call, one parse, no retry/fallback yet
(that's a later step). If the model returns invalid JSON or a value that
fails Pydantic validation, this raises LLMResponseError and the caller
decides what to do.
"""

import json
import os
import time

from dotenv import load_dotenv
from groq import Groq
from pydantic import ValidationError

from app.schemas import ErrorCategory, IncidentAnalysis, Priority

load_dotenv()

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set - copy .env to .env and fill it in."
            )
        _client = Groq(api_key=api_key)
    return _client


class LLMResponseError(Exception):
    """Raised when the model's response can't be parsed into IncidentAnalysis."""


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


def analyze_incident(
    service: str, environment: str, normalized_text: str
) -> tuple[IncidentAnalysis, int]:
    """
    Single, unretried call to Groq. Returns (analysis, latency_ms).
    Raises LLMResponseError on invalid/unparseable output.
    """
    client = _get_client()
    started = time.monotonic()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(service, environment, normalized_text)},
        ],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    latency_ms = int((time.monotonic() - started) * 1000)

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMResponseError(f"Model did not return valid JSON: {e}\nRaw: {content!r}") from e

    try:
        analysis = IncidentAnalysis(**data)
    except ValidationError as e:
        raise LLMResponseError(f"Model JSON failed schema validation: {e}\nRaw: {data!r}") from e

    return analysis, latency_ms
