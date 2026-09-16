from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime
from uuid import UUID


class ErrorCategory(str, Enum):
    DATABASE_TIMEOUT = "database_timeout"
    NULL_POINTER = "null_pointer"
    RATE_LIMIT = "rate_limit_exceeded"
    AUTH_FAILURE = "auth_failure"
    NETWORK_PARTIAL_FAILURE = "network_partial_failure"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# --- INPUT ---

class Environment(str, Enum):
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"

    @classmethod
    def _missing_(cls, value: object) -> "Environment | None":
        """
        Fallback used by Python's Enum (and, through it, Pydantic's enum
        validation) whenever `value` doesn't match a member directly - lets
        us accept common long-form aliases without touching call sites or
        adding a separate field validator.
        """
        if not isinstance(value, str):
            return None
        aliases = {"development": cls.DEV, "production": cls.PROD}
        return aliases.get(value.lower())

class RawLogEntry(BaseModel):
    service: str
    environment: Environment
    timestamp: datetime
    raw_text: str = Field(..., description="Raw stack trace or error log")


class BatchAnalyzeRequest(BaseModel):
    entries: list[RawLogEntry]


# --- OUTPUT (LLM returns) ---
#
# Task decomposition: classification and prioritization are two separate
# LLM calls (see app/llm_client.py) - ClassificationResult and
# PriorityResult are what each step returns on its own, and IncidentAnalysis
# below is the two merged into the single shape the rest of the app (and
# the API response) has always used. Splitting these out, rather than one
# call doing everything, keeps each prompt focused on one judgment.

class ClassificationResult(BaseModel):
    category: ErrorCategory
    root_cause_summary: str = Field(..., max_length=300)
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_human_review: bool


class PriorityResult(BaseModel):
    priority: Priority
    priority_reasoning: str = Field(..., max_length=200)


class IncidentAnalysis(BaseModel):
    category: ErrorCategory
    root_cause_summary: str = Field(..., max_length=300)
    priority: Priority
    priority_reasoning: str = Field(..., max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_human_review: bool


# --- saving to the db / returns to the client ---

class IncidentRecord(BaseModel):
    id: Optional[UUID] = None
    service: str
    environment: str
    timestamp: datetime
    raw_text_hash: str
    analysis: IncidentAnalysis
    is_duplicate: bool = False
    duplicate_of_id: Optional[UUID] = None
    occurrence_count: int = 1
    llm_retry_count: int = 0
    llm_latency_ms: Optional[int] = None


class BatchAnalyzeResponse(BaseModel):
    results: list[IncidentRecord]
