"""
Trace of one analysis: every model call that went into it, attempt by attempt.

The incidents table only keeps the final answer plus llm_retry_count and
llm_latency_ms. When the model gets an incident wrong, that's not enough to
see why - you want the exact prompt, the raw response, and what went wrong on
each failed attempt. That record is what these models describe; it's saved
to MongoDB (see app/mongo.py) as one document per analysis.

Its shape varies by design: single-call mode has one step, decomposed mode
two, and each step has 1..LLM_MAX_ATTEMPTS attempts - a natural fit for one
nested document rather than several joined tables.

Statuses are plain Literal strings rather than Enums so they go into BSON
as-is (pymongo encodes str, not str-Enum members).
"""

import hashlib
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, computed_field

AttemptStatus = Literal[
    "ok",
    "invalid_json",  # response wasn't JSON at all
    "validation_error",  # JSON, but failed the expected schema
    "rate_limited",
    "timeout",
    "connection_error",
    "server_error",  # Groq 5xx
]

AnalysisMode = Literal["single", "decomposed"]


def prompt_fingerprint(system_prompt: str) -> str:
    """
    Short hash identifying a system prompt. The system prompts are static,
    so storing their full text in every trace would duplicate the same few KB
    thousands of times - the fingerprint is enough to tell which prompt
    version produced a trace, and it changes by itself whenever a prompt is
    edited, with no version number to remember to bump.
    """
    return hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:12]


class AttemptTrace(BaseModel):
    """One request to the model."""

    n: int  # 1-based
    status: AttemptStatus
    latency_ms: int  # this request only, not the backoff wait before it
    error: str | None = None
    # What the model actually returned - present whenever a response came
    # back (ok, invalid_json, validation_error), absent for API errors.
    raw_response: str | None = None


class StepTrace(BaseModel):
    """One pipeline step (analyze, classify or prioritize) and its attempts."""

    name: str
    system_prompt_fingerprint: str
    # Holds the (normalized) raw log text, which may contain secrets or
    # personal data from a stack trace - one reason traces expire.
    user_prompt: str
    attempts: list[AttemptTrace] = Field(default_factory=list)
    # The parsed result the step settled on, or the fallback if it fell back.
    result: dict | None = None
    fell_back: bool = False
    latency_ms: int = 0  # whole step, backoff waits included

    @computed_field
    @property
    def retry_count(self) -> int:
        return max(len(self.attempts) - 1, 0)


class LLMRun(BaseModel):
    """
    Every model call behind one IncidentAnalysis. Totals are computed fields,
    so they are stored in the document too and can be filtered/aggregated on
    directly (e.g. find all runs that fell back).
    """

    mode: AnalysisMode
    model: str
    steps: list[StepTrace]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @computed_field
    @property
    def latency_ms(self) -> int:
        return sum(s.latency_ms for s in self.steps)

    @computed_field
    @property
    def retry_count(self) -> int:
        return sum(s.retry_count for s in self.steps)

    @computed_field
    @property
    def fell_back(self) -> bool:
        return any(s.fell_back for s in self.steps)
