# AI Log Incident Analyzer

A service that takes raw, messy application error logs / stack traces and
turns them into structured, prioritized incidents — automatic root-cause
summaries and triage priority for on-call engineers, with the kind of
idempotency and observability concerns a real ingestion pipeline needs.

Input is deliberately messy: stack traces in different formats (Python
tracebacks, Java/Kotlin exceptions, Go panics, generic log lines), plus
service/environment context. Output is structured and validated (Pydantic):
error category, plain-language root cause, priority with reasoning,
confidence, and a `needs_human_review` flag for anything ambiguous.

## How it works

```
POST /analyze-incident
   │
   ▼
normalize_raw_text()      — trim the trace to message + top N frames
   │
   ▼
compute_error_hash()      — hash of the normalized text (dedup key)
   │
   ▼
find_active_incident()    — seen this hash in the last N minutes?
   │
   ├── yes → record_duplicate()      → bump occurrence_count, NO LLM call
   │
   └── no  → analyze_incident()      → real Groq LLM call
             record_new_incident()   → insert new row
   │
   ▼
IncidentRecord (JSON response)
```

The dedup check runs *before* the LLM call and lives in Postgres rather
than an in-process cache. That matters for two real reasons: an in-memory
dict wouldn't survive a restart, and it wouldn't be shared across multiple
worker processes — both would silently let duplicate bursts re-trigger LLM
calls. A hit within the dedup window (default 10 minutes) just bumps a
counter on the existing row; once the window expires, the same error
hash is treated as a fresh incident (worth a new analysis, since a bug
resurfacing after a real gap may have a different cause).

## Tech stack

- **FastAPI** + **Pydantic** — API and schema validation
- **Groq** (`openai/gpt-oss-120b` by default) — LLM classification, JSON mode
- **PostgreSQL 18** + **SQLAlchemy (async)** + **asyncpg** — incident storage & dedup
- **Docker Compose** — app + database, wired together

## Project structure

```
app/
  main.py          FastAPI app: parse -> dedup check -> LLM (if needed) -> response
  schemas.py       Pydantic models (input, LLM output, stored record)
  parser.py        raw log normalization + hashing
  llm_client.py    Groq integration
  db.py            async SQLAlchemy engine/session + Incident table
  dedup.py         dedup window lookup + counter updates
data/
  synthetic_logs.py   generates 40+ labeled synthetic log examples (ground truth
                       for future eval), covering all error categories across
                       Python/Java/Go/gRPC formats and prod/staging/dev
Dockerfile
docker-compose.yml   app + Postgres, wired together
```

## Running it

### Docker Compose (recommended)

```bash
docker compose up -d --build
```

This builds the app image and starts Postgres + the API together — the
app waits for the database to report healthy before starting. Postgres
data persists in a named volume (`pgdata`) across restarts.

```bash
curl http://localhost:8000/health
```

To stop:

```bash
docker compose down        # keeps data
docker compose down -v     # also wipes the database volume
```

### Running the app locally (without Docker)

Requires a Postgres instance reachable at the `DATABASE_URL` in `.env`
(e.g. just run `docker compose up -d db` for the database only):

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Configuration

Copy `.env.example` to `.env` and fill in:

| Variable               | Purpose                                              | Default (local)                                         |
|------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `GROQ_API_KEY`         | Groq API key (free tier at console.groq.com)          | —                                                           |
| `GROQ_MODEL`           | Groq model to use                                     | `openai/gpt-oss-120b`                                      |
| `DATABASE_URL`         | Postgres connection string (asyncpg driver)           | `postgresql+asyncpg://root:root@localhost:5432/incident_analyzer` |
| `DEDUP_WINDOW_MINUTES` | How long a repeat error is treated as a duplicate     | `10`                                                        |

Inside `compose.yaml`, `DATABASE_URL` is overridden to point at the
`db` service hostname instead of `localhost`, since that's how containers
reach each other on the compose network.

## API

### `POST /analyze-incident`

Request:

```json
{
  "service": "payments-api",
  "environment": "prod",
  "timestamp": "2026-09-12T20:00:00Z",
  "raw_text": "Traceback (most recent call last):\n  File \"app/db/session.py\", line 42, in get_connection\n    conn = pool.acquire(timeout=5)\npsycopg2.OperationalError: timeout expired"
}
```

Response (first time this error is seen — real LLM call):

```json
{
  "id": 1,
  "service": "payments-api",
  "environment": "prod",
  "timestamp": "2026-09-12T20:00:00Z",
  "raw_text_hash": "15f4a0cd...",
  "analysis": {
    "category": "database_timeout",
    "root_cause_summary": "A database operation timed out while acquiring a connection, causing payment processing to fail.",
    "priority": "critical",
    "priority_reasoning": "Database timeout in the production payments service can block transactions and impact revenue.",
    "confidence": 0.95,
    "needs_human_review": false
  },
  "is_duplicate": false,
  "duplicate_of_id": null,
  "occurrence_count": 1,
  "llm_retry_count": 0,
  "llm_latency_ms": 1036
}
```

Sending the *same* `raw_text` again within the dedup window returns
`is_duplicate: true`, `duplicate_of_id` pointing at the original incident,
and an incremented `occurrence_count` — without another LLM call.

### `GET /health`

Basic liveness check, returns `{"status": "ok"}`.

## Notable design decisions

- **Category is an enum, not free text** — if the model returns anything
  outside the known set, Pydantic validation catches it immediately
  instead of letting the model invent categories.
- **`confidence` vs `needs_human_review` are separate fields** —
  `confidence` is a number for sorting/analytics; `needs_human_review` is
  meant to be an explicit decision, not left entirely to the model's
  self-assessment.
- **Truncation keeps the message + first 5 stack frames** — deep
  framework/stdlib frames add tokens without adding diagnostic signal;
  the frames closest to the failure point almost always carry the actual
  clue.
- **Dedup hashes the normalized text, not the raw text** — so cosmetic
  differences (timestamps, whitespace, line order) don't produce
  different hashes for what is semantically the same recurring error.
- **One row per distinct incident, not per occurrence** — duplicates
  update `occurrence_count`/`last_seen_at` on the existing row rather than
  inserting a new one, which is what keeps the "don't call the LLM 500
  times for the same burst" property cheap to enforce with an indexed
  lookup.
