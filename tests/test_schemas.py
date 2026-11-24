import pytest
from pydantic import ValidationError

from app.schemas import Environment, ErrorCategory, IncidentAnalysis, RawLogEntry

VALID_ANALYSIS = {
    "category": "database_timeout",
    "root_cause_summary": "Connection pool exhausted.",
    "priority": "high",
    "priority_reasoning": "Checkout is affected.",
    "confidence": 0.9,
    "needs_human_review": False,
}


@pytest.mark.parametrize("value, expected", [
    ("dev", Environment.DEV),
    ("staging", Environment.STAGING),
    ("prod", Environment.PROD),
    ("development", Environment.DEV),
    ("Production", Environment.PROD),
])
def test_environment_accepts_values_and_long_form_aliases(value, expected):
    assert Environment(value) is expected


@pytest.mark.parametrize("value", ["qa", "", "produccion", 3])
def test_environment_rejects_anything_else(value):
    with pytest.raises(ValueError):
        Environment(value)


def test_raw_log_entry_parses_aliases_and_timestamps():
    entry = RawLogEntry(service="api", environment="production", timestamp="2026-11-24T19:00:00Z", raw_text="boom")
    assert entry.environment is Environment.PROD
    assert entry.timestamp.year == 2026


@pytest.mark.parametrize("missing", ["service", "environment", "timestamp", "raw_text"])
def test_raw_log_entry_requires_every_field(missing):
    data = {"service": "api", "environment": "prod", "timestamp": "2026-11-24T19:00:00Z", "raw_text": "boom"}
    del data[missing]
    with pytest.raises(ValidationError):
        RawLogEntry(**data)


def test_incident_analysis_accepts_a_valid_reply():
    analysis = IncidentAnalysis(**VALID_ANALYSIS)
    assert analysis.category is ErrorCategory.DATABASE_TIMEOUT


@pytest.mark.parametrize("field, bad_value", [
    ("category", "made_up"),
    ("priority", "urgent"),
    ("confidence", 1.5),
    ("confidence", -0.1),
    ("root_cause_summary", "x" * 301),
    ("priority_reasoning", "x" * 201),
])
def test_incident_analysis_rejects_out_of_contract_values(field, bad_value):
    with pytest.raises(ValidationError):
        IncidentAnalysis(**{**VALID_ANALYSIS, field: bad_value})
