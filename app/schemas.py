from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum
from datetime import datetime


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

class RawLogEntry(BaseModel):
    service: str
    environment: str  # prod / staging / dev
    timestamp: datetime
    raw_text: str = Field(..., description="Raw stack trace or error log")


class BatchAnalyzeRequest(BaseModel):
    entries: list[RawLogEntry]


# --- OUTPUT (LLM returns) ---

class IncidentAnalysis(BaseModel):
    category: ErrorCategory
    root_cause_summary: str = Field(..., max_length=300)
    priority: Priority
    priority_reasoning: str = Field(..., max_length=200)
    confidence: float = Field(..., ge=0.0, le=1.0)
    needs_human_review: bool


# --- saving to the db / returns to the client ---

class IncidentRecord(BaseModel):
    id: Optional[int] = None
    service: str
    environment: str
    timestamp: datetime
    raw_text_hash: str
    analysis: IncidentAnalysis
    is_duplicate: bool = False
    duplicate_of_id: Optional[int] = None
    occurrence_count: int = 1
    llm_retry_count: int = 0
    llm_latency_ms: Optional[int] = None


class BatchAnalyzeResponse(BaseModel):
    results: list[IncidentRecord]
