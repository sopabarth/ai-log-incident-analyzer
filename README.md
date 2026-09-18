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
   └── no  → analyze_incident()      → real Groq call(s) (retry/fallback, task decomposition - see below)
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

## Retry & fallback

A single bad LLM response (invalid JSON, or JSON that fails schema
validation) or a transient Groq API hiccup shouldn't fail the whole
request. `analyze_incident()` retries up to `LLM_MAX_ATTEMPTS` times
(default 3), with different handling per failure type:

| Failure                                                              | Behavior                                              |
|-----------------------------------------------------------------------|--------------------------------------------------------|
| Malformed output (bad JSON / fails `IncidentAnalysis` validation)      | Retry immediately — it's a one-off bad generation, not a timing issue |
| Transient API error (connection drop, timeout, rate limit, 5xx)       | Retry with a short backoff (`LLM_RETRY_BACKOFF_SECONDS × attempt`) — hammering a struggling/throttled API immediately tends to make it worse |
| Both of the above, still failing after all attempts                   | **Fall back** to a generic result instead of failing the request |
| Non-retryable (bad/missing API key, malformed request, permission denied) | Propagate immediately — not retried, not papered over |

The fallback result is deliberately not a guess:

```json
{
  "category": "unknown",
  "root_cause_summary": "Automated analysis failed after repeated attempts - the model did not return a usable classification for this error.",
  "priority": "medium",
  "priority_reasoning": "Priority could not be determined automatically; needs manual triage.",
  "confidence": 0.0,
  "needs_human_review": true
}
```

`confidence: 0.0` and `needs_human_review: true` make it unmistakable that
this wasn't a real classification, and `priority: medium` is a deliberate
middle ground — `low` risks a genuinely serious incident being ignored,
`critical` risks paging someone over what might just be a bad model run.

Non-retryable errors (a missing/invalid `GROQ_API_KEY`, a malformed
request, etc.) are intentionally **not** retried or hidden behind the
fallback — retrying a guaranteed-to-fail auth error just burns time, and
silently returning "unknown incident" for a broken deployment would mask
an ops problem that needs to surface loudly, not get quietly classified.

The response's `llm_retry_count` reflects how many attempts beyond the
first were needed — `0` means it succeeded on the first try.

## Task decomposition

`analyze_incident()` doesn't ask one prompt to do everything. It's two
focused Groq calls:

1. **Classify** — category, root cause summary, confidence, needs_human_review
2. **Prioritize** — given the category from step 1 plus the same
   service/environment/error text, just the priority + reasoning

Each step is retried/falls back independently, using the same mechanism
described above.

**Why priority still needs its own LLM call instead of a deterministic
`(category, environment) → priority` lookup table** (the first, simpler
idea): checking that against `data/synthetic_logs.py`'s own ground truth
shows the same category in the same environment legitimately spans
multiple priorities — e.g. `auth_failure` in `prod` is labeled `critical`,
`high`, *and* `low` across different examples, depending entirely on the
blast radius described in the error text ("all requests rejected" vs "one
user locked out"). A lookup table keyed on category+environment alone
would have regressed accuracy, not just simplified the code - so priority
keeps reading the actual error text, just as a separate, focused call
rather than bundled into classification.

**`TASK_DECOMPOSITION` env var** (default `false`) switches between a
single combined call that asks for everything at once - the way this
worked before decomposition - and this two-call pipeline. Both paths
share the exact same retry/fallback machinery; the toggle exists to let
the two approaches be compared directly rather than deleting the simpler
one.
Confirmed on the same input, both approaches can disagree - see Eval
results below for a full head-to-head comparison, not just one example.

## Eval results

`eval/run_eval.py` runs every example in `data/synthetic_logs.json` (41
hand-labeled logs) through the real pipeline - actual Groq calls, no
mocking - and scores the output against the ground truth: category
accuracy/precision/recall, priority exact-match plus "how far off" it was,
and `needs_human_review` accuracy. It bypasses FastAPI/dedup/Postgres on
purpose, since this measures model+pipeline quality, not the HTTP/DB
plumbing.

```bash
python -m eval.run_eval                # uses the current TASK_DECOMPOSITION setting
python -m eval.run_eval --mode both     # run every example through both pipelines, compare
python -m eval.run_eval --limit 5       # smoke test on a subset
python -m eval.run_eval --save          # also write raw results under eval/results/
```

Full run, both modes, all 41 examples:

| Metric                          | Single-call | Decomposed |
|----------------------------------|:-----------:|:----------:|
| Category accuracy                | 95.1% (39/41) | **97.6%** (40/41) |
| Priority exact-match              | 63.4%       | **75.6%**  |
| Priority mean distance (0 = exact) | 0.37      | **0.29**   |
| `needs_human_review` accuracy     | 97.6%       | 97.6%      |

Decomposition won on both category and priority accuracy, most visibly on
priority (75.6% vs 63.4% exact-match) - consistent with the reasoning
above: a focused second call that only judges priority, with the category
already fixed, outperforms asking for everything in one shot. The two
modes disagreed on 14 of the 41 examples; category misses in both modes
were the same genuinely ambiguous case (a log with "No indication of what
operation or why it failed", expected `unknown`, both modes guessed
`network_partial_failure`).

This same full run also exercised the retry/fallback path for real: one
example hit an actual `APIConnectionError`, retried per policy, and fell
back cleanly (`unknown`/`medium`, `confidence: 0.0`) instead of crashing
the run - and, run back-to-back at 41-82 requests, later calls visibly
slowed down (up to ~11s vs ~700-1000ms for the first ones), consistent
with Groq's free tier softly throttling under sustained volume before
ever returning a hard `RateLimitError`. Neither is a bug in the eval
script; both are exactly the kind of behavior retry/fallback exists to
absorb.

## Tech stack

- **FastAPI** + **Pydantic** — API and schema validation
- **Groq** (`openai/gpt-oss-120b` by default) — LLM classification, JSON mode
- **PostgreSQL 18** + **SQLAlchemy (async)** + **asyncpg** — incident storage & dedup
- **Alembic** — schema migrations
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
  synthetic_logs.py   generates 40+ labeled synthetic log examples (ground
                       truth for eval/run_eval.py), covering all error
                       categories across Python/Java/Go/gRPC formats and
                       prod/staging/dev
eval/
  run_eval.py          scores the real pipeline against synthetic_logs.json -
                        category/priority accuracy, precision/recall, see
                        Eval results above
alembic/             migration environment (env.py reuses app.db's DATABASE_URL)
alembic/versions/    one file per migration, applied in order
Dockerfile
compose.yaml         app + Postgres, wired together
```

### Schema migrations (Alembic)

The `Incident.id` primary key is a UUID (`uuid.uuid4`, generated in Python),
not an auto-incrementing int — it doesn't leak how many incidents exist or
their creation order, and avoids collisions if incidents are ever inserted
from more than one place without a shared sequence.

Schema is entirely Alembic-owned - the app no longer runs `create_all()`
at startup. Inside Docker, the app container runs `alembic upgrade head`
automatically before starting uvicorn (see `Dockerfile`). Locally:

```bash
alembic upgrade head                                  # apply migrations
alembic revision --autogenerate -m "describe the change"   # after editing app/db.py models
```

`alembic/env.py` reads `DATABASE_URL` from the same place `app/db.py`
does (the environment / `.env`), so there's one connection string to keep
in sync, not two.

### Enum enforcement at the database level

`category`, `priority`, and `environment` are validated by Pydantic on the
way in, but nothing stopped a value outside that set from reaching the
database through any other path (a manual `psql` fix during an incident
postmortem, a future script, a bug that bypasses the schema). Two
different mechanisms close that gap, chosen per-column based on how often
each taxonomy is expected to change:

- **`environment` and `priority` are native Postgres `ENUM` types**
  (`environment_enum`, `priority_enum`) — these lists (`dev`/`staging`/`prod`,
  `critical`/`high`/`medium`/`low`) aren't expected to grow.
- **`category` is a plain `VARCHAR` with a `CHECK` constraint**
  (`ck_incidents_category`) instead — error categories are the one
  taxonomy here likely to expand over time, and evolving a native Postgres
  enum is real migration pain: `ALTER TYPE ... ADD VALUE` can't be used in
  the same transaction it runs in, and renaming/removing a value isn't
  supported at all (the type has to be recreated). A `CHECK` constraint
  gives the same "no garbage values" guarantee at the database level and
  is a plain `DROP CONSTRAINT` / `ADD CONSTRAINT` to change.

Both were confirmed to actually reject bad data at the database layer, not
just in the app - a raw `UPDATE ... SET environment = 'staging_v2'` and a
raw `UPDATE ... SET category = 'made_up_category'` both fail with a
`psql` error.

One Alembic gotcha worth knowing if you touch this: `--autogenerate`
detected the native-enum type changes on `environment`/`priority` fine,
but silently produced **nothing** for the `category` `CHECK` constraint
(it doesn't diff `CheckConstraint`s), and the enum-column migration it did
generate omits the `CREATE TYPE` step entirely (`alter_column` assumes the
Postgres type already exists). Both had to be added by hand in the
migration file - autogenerate output should always be read, not applied
blindly, and this is a concrete example of where it falls short.

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
| `LLM_MAX_ATTEMPTS`     | Max attempts before falling back (see Retry & fallback) | `3`                                                        |
| `LLM_RETRY_BACKOFF_SECONDS` | Backoff multiplier between retries on transient API errors | `0.5`                                            |
| `TASK_DECOMPOSITION`   | `true`: classify + prioritize as two calls; `false`: one combined call | `false`                                      |

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
- **Retryable failures vs. non-retryable failures are handled differently**
  — malformed output and transient API errors get retried (then a safe
  fallback if still failing); a broken deployment (bad API key, auth
  failure) fails loudly instead of quietly turning into an "unknown"
  incident, since papering over a config problem would hide it from
  whoever needs to fix it.
- **Priority is decomposed into its own LLM call, not a lookup table** —
  verified against the ground-truth data first: the same category in the
  same environment genuinely spans multiple priorities depending on blast
  radius described in the error text, so priority keeps reading that text
  rather than being inferred from category+environment alone.
