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
   │
   ▼ (background, after the response is sent; fresh analyses only)
save_llm_run()            — trace of every model call → MongoDB llm_runs
```

The dedup check runs *before* the LLM call and lives in Postgres rather
than an in-process cache. That matters for two real reasons: an in-memory
dict wouldn't survive a restart, and it wouldn't be shared across multiple
worker processes — both would silently let duplicate bursts re-trigger LLM
calls. A hit within the dedup window (default 10 minutes) just bumps a
counter on the existing row; once the window expires, the same error
hash is treated as a fresh incident.

## Retry & fallback

A single bad LLM response (invalid JSON, or JSON that fails schema
validation) or a transient Groq API hiccup shouldn't fail the whole
request. `analyze_incident()` retries up to `LLM_MAX_ATTEMPTS` times (default 3), with different handling per failure
type:

| Failure                                                                   | Behavior                                                                                                                                     |
|---------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| Malformed output (bad JSON / fails `IncidentAnalysis` validation)         | Retry immediately — it's a one-off bad generation, not a timing issue                                                                        |
| Transient API error (connection drop, timeout, rate limit, 5xx)           | Retry with a short backoff (`LLM_RETRY_BACKOFF_SECONDS × attempt`) — hammering a struggling/throttled API immediately tends to make it worse |
| Both of the above, still failing after all attempts                       | **Fall back** to a generic result instead of failing the request                                                                             |
| Non-retryable (bad/missing API key, malformed request, permission denied) | Propagate immediately — not retried, not papered over                                                                                        |

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
uv run python -m eval.run_eval          # uses the current TASK_DECOMPOSITION setting
uv run python -m eval.run_eval --mode both # run every example through both pipelines, compare
uv run python -m eval.run_eval --limit 5   # smoke test on a subset
uv run python -m eval.run_eval --save      # also write raw results under eval/results/
```

Full run, both modes, all 41 examples:

| Metric                             |  Single-call  |    Decomposed     |
|------------------------------------|:-------------:|:-----------------:|
| Category accuracy                  | 95.1% (39/41) | **97.6%** (40/41) |
| Priority exact-match               |     63.4%     |     **75.6%**     |
| Priority mean distance (0 = exact) |     0.37      |     **0.29**      |
| `needs_human_review` accuracy      |     97.6%     |       97.6%       |

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

## LLM call traces (MongoDB)

The incidents table keeps only the final answer plus `llm_retry_count` and
`llm_latency_ms`. When the model gets an incident wrong, that isn't enough
to see why. So every fresh analysis also writes one document to the
MongoDB collection `llm_runs`: which mode and model ran, and for each step
the user prompt, every attempt (status, error, latency, the model's raw
response) and the result it settled on.

```json
{
  "source": "api",
  "incident_id": "3f2a…",
  "error_hash": "15f4a0cd…",
  "service": "payments-api",
  "environment": "prod",
  "mode": "decomposed",
  "model": "openai/gpt-oss-120b",
  "created_at": "2026-11-24T19:00:00Z",
  "steps": [
    {
      "name": "classify",
      "system_prompt_fingerprint": "9c1e0b7a42d3",
      "user_prompt": "service: payments-api\nenvironment: prod\nerror:\n…",
      "attempts": [
        {"n": 1, "status": "validation_error", "latency_ms": 820,
         "error": "schema validation failed: category: Input should be …", "raw_response": "{…}"},
        {"n": 2, "status": "ok", "latency_ms": 760, "raw_response": "{…}"}
      ],
      "result": {"category": "database_timeout", "confidence": 0.95, "…": "…"},
      "fell_back": false, "latency_ms": 1580, "retry_count": 1
    },
    {"name": "prioritize", "…": "…"}
  ],
  "latency_ms": 2220, "retry_count": 1, "fell_back": false
}
```

Attempt statuses: `ok`, `invalid_json`, `validation_error`, `rate_limited`,
`timeout`, `connection_error`, `server_error`.

**Why a second database, and why a document store.** Incidents stay in
Postgres: they need constraints, database-level enum validation and the
indexed dedup lookup. Traces are different data: nested and varying in
shape (one step or two, one to three attempts each), append-only, and
short-lived. That is a natural fit for one document per analysis.
Postgres could hold them too, as a `JSONB` column, and for a project this
size that would be a legitimate choice. The document store earns its place
with a built-in TTL for expiry, schema changes that need no migrations, and
the aggregation pipeline for querying across traces.

Design decisions:

- **Best effort.** A trace is diagnostics, not business data, so a Mongo
  outage must not fail an analysis that already succeeded. The write runs as
  a FastAPI background task after the response is sent (no added latency),
  errors are caught and printed, and the client gives up after 2 s instead
  of pymongo's default 30 s.
- **Postgres first, then Mongo.** There is no transaction spanning both
  databases. Writing the incident first means a failed trace write leaves an
  incident without a trace, never a trace pointing at an incident that
  doesn't exist.
- **Only real model calls are traced.** A dedup hit calls no model, so it
  writes no trace. A fallback does get one, since that is the case you most
  want to inspect.
- **The system prompt is stored as a fingerprint.** It's a short SHA-256 of
  the prompt text, not the text itself. The system prompts are static, so
  storing them would repeat the same few KB in every document. The
  fingerprint still tells which prompt version produced a trace, and it
  changes by itself when a prompt is edited.
- **Traces contain raw log text.** The user prompt and raw responses can
  carry secrets or personal data from a stack trace. That is a reason to give
  them a retention limit: they expire after `LLM_TRACE_TTL_DAYS` (see Indexes).
- **Non-retryable errors leave no trace.** A bad API key, for example,
  propagates as an exception and no incident is created.
- `source` is `"api"` for now. Eval runs will be `"eval"`, so statistics can
  tell them apart from real traffic. Eval traces have no incident, so they
  leave `incident_id` out rather than storing `null` (see below).

`AsyncMongoClient` (pymongo's own async API; Motor is deprecated) is created
in the FastAPI lifespan, because an async client belongs to the event loop
it was created in. It's configured with `uuidRepresentation="standard"`:
pymongo has no default and refuses to encode a `uuid.UUID` without one.

The first trace written already paid for itself: it showed the API sending
the model `environment: Environment.PROD` instead of `environment: prod`.
Since Python 3.12, an f-string renders a `(str, Enum)` member by its name,
and the endpoint passed the enum where `analyze_incident()` expects a
string. The eval never hit this because it reads environments as plain
strings from JSON, so its numbers weren't affected. Neither the API response
nor the incidents table could have shown it; only the exact prompt could.

Looking at traces (`docker compose exec mongo mongosh -u root -p root`):

```js
use incident_analyzer
db.llm_runs.find().sort({created_at: -1}).limit(1)
db.llm_runs.find({fell_back: true})
db.llm_runs.find({"steps.attempts.status": "validation_error"}, {service: 1, "steps.attempts": 1})
```

### Indexes

MongoDB has no schema to migrate, so there is no Alembic counterpart here.
Instead, `ensure_indexes()` runs at app startup and declares the indexes
`llm_runs` should have. Creating an index that already exists with the same
options does nothing, so it is safe on every start. It's best effort like
the writes: if Mongo is down, the app starts anyway and the indexes are
created on the next start.

| Index                   | Keys                               | Options                                         | Serves                                      |
|-------------------------|------------------------------------|-------------------------------------------------|---------------------------------------------|
| `incident_id_unique`    | `incident_id`                      | unique, partial: `{incident_id: {$exists: true}}` | the trace of an incident                    |
| `error_hash_created_at` | `error_hash` asc, `created_at` desc | -                                               | every trace of one error, newest first      |
| `created_at_ttl`        | `created_at`                       | `expireAfterSeconds` = `LLM_TRACE_TTL_DAYS` × 86400 | automatic expiry                            |

- **`incident_id` is unique.** An incident is analyzed exactly once, so a
  second trace for one would be a bug, and the database rejects it (E11000).
  It's *partial* because eval traces have no incident, and a plain unique
  index would let only one document lack the value. One detail that was
  verified with `explain()`, not assumed: the first version filtered on
  `{incident_id: {$type: "binData"}}`, and lookups by `incident_id` were full
  collection scans (`COLLSCAN`). The planner uses a partial index only when
  it can prove the query falls inside the filter, and it proves that
  `{incident_id: <uuid>}` implies `$exists: true` but not `$type`. With
  `$exists` the same lookup is an index scan. `$exists` counts an explicit
  `null` as present, which is why eval traces omit the field.
- **Compound `error_hash` + `created_at`.** One index serves both the filter
  and the sort, so Mongo reads only the matching entries, already in order
  (`IXSCAN`, one key and one document examined for one match), instead of
  sorting in memory.
- **TTL.** A background monitor deletes documents older than the TTL about
  once a minute, so expiry isn't to the second. It only works on a BSON date,
  which is why `created_at` is stored as a `datetime`, not an ISO string.
  `create_index()` can't change an existing index's options (it fails with
  `IndexOptionsConflict`), so when `LLM_TRACE_TTL_DAYS` changes, the next
  start applies the new value in place with `collMod`. Other index changes,
  like the partial filter above, can't be done with `collMod`: drop the old
  index once (`db.llm_runs.dropIndex("<name>")`) and restart the app.

Checking that a query uses an index:

```js
db.llm_runs.find({error_hash: "<hash>"}).sort({created_at: -1}).explain("executionStats")
// winningPlan: FETCH <- IXSCAN (error_hash_created_at), not COLLSCAN
```

### Statistics (aggregation pipeline)

`GET /stats/llm` (see API below) answers questions such as "how often does
each step fail, and how" or "how slow is decomposed mode compared to single
call" from the stored traces. MongoDB computes the answer itself, with one
aggregation pipeline (`app/llm_stats.py`). A pipeline is a list of stages,
each transforming the documents the previous stage produced, a bit like a
chain of SQL clauses in an explicit order:

```
$match   created_at >= now - days, source = "api"      (WHERE; uses an index)
   │
$facet ──┬── overview:       $group (all runs)          → runs, fallbacks, retries, latency
         ├── by_mode_model:  $group by {mode, model}    → the same, per mode/model
         └── steps:          $unwind steps
                             $unwind steps.attempts     → one document per attempt
                             $group by {step, status}   → attempts per status
                             $group by step             → totals, failure rate,
                                                          statuses as {"ok": 5, ...}
```

- **`$facet`** runs three sub-pipelines over the same matched documents and
  returns all three results in one document: one round trip instead of three
  queries.
- **`$match` comes first, outside `$facet`,** because stages inside `$facet`
  never use indexes. As the first stage it uses the TTL index on `created_at`
  as an ordinary index for the date range (`IXSCAN(created_at_ttl)` in
  `explain`). The planner also works out which fields the rest of the
  pipeline needs and reads only those.
- **`$unwind` twice** makes nested data groupable. A decomposed run with 2
  attempts at `classify` and 1 at `prioritize` becomes three documents, one
  per attempt. In Postgres this would be a join of runs, steps and attempts
  tables, or `jsonb_array_elements` twice.
- **Two `$group`s in a row** first count attempts per (step, status), then
  roll those counts up per step. `$arrayToObject` turns the collected
  `{k, v}` pairs into a `statuses` object.
- **`$percentile`** (MongoDB 7.0+) gives p50/p95 latency. Inside `$group` only
  the `approximate` method is available. On small sets it returns one of the
  actual values rather than interpolating between two. With the four real
  traces in the example below, p50 is 954 ms, which is one of the four
  latencies, not the midpoint of the middle two.

The pipeline was checked against hand-built traces with hand-computed
expected values (counts, rates, means, the per-status breakdown), plus
traces that the date and source filters must exclude.

## Tech stack

- **FastAPI** + **Pydantic** — API and schema validation
- **Groq** (`openai/gpt-oss-120b` by default) — LLM classification, JSON mode
- **PostgreSQL 18** + **SQLAlchemy (async)** + **asyncpg** — incident storage & dedup
- **Alembic** — schema migrations
- **MongoDB 8** + **pymongo** (`AsyncMongoClient`) — LLM call traces
- **uv** — dependency management (`pyproject.toml` + `uv.lock`)
- **Docker Compose** — app + both databases, wired together

## Project structure

```
app/
  main.py          FastAPI app: parse -> dedup check -> LLM (if needed) -> response
  schemas.py       Pydantic models (input, LLM output, stored record)
  parser.py        raw log normalization + hashing
  llm_client.py    Groq integration
  llm_trace.py     Pydantic models for the per-analysis trace of model calls
  mongo.py         MongoDB client, indexes, saving traces to llm_runs
  llm_stats.py     aggregation pipeline behind GET /stats/llm
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
pyproject.toml         dependencies (replaces requirements.txt)
uv.lock                exact, reproducible dependency versions - commit this
requirements.txt       same dependencies for the pip-based image (Dockerfile.pip)
Dockerfile.uv          app image built with uv (default)
Dockerfile.pip         equivalent app image built with pip + requirements.txt
compose.yaml         app + Postgres + MongoDB, wired together
```

### Schema migrations (Alembic)

The `Incident.id` primary key is a UUID (`uuid.uuid4`, generated in Python),
not an auto-incrementing int — it doesn't leak how many incidents exist or
their creation order, and avoids collisions if incidents are ever inserted
from more than one place without a shared sequence.

Schema is entirely Alembic-owned - the app no longer runs `create_all()`
at startup. Inside Docker, the app container runs `alembic upgrade head`
automatically before starting the server (see `Dockerfile.uv` / `Dockerfile.pip`). Locally:

```bash
uv run alembic upgrade head                                  # apply migrations
uv run alembic revision --autogenerate -m "describe the change"   # after editing app/db.py models
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
but silently produced **nothing** for the `category` `CHECK` constraint (it doesn't diff `CheckConstraint`s), and the
enum-column migration it did
generate omits the `CREATE TYPE` step entirely (`alter_column` assumes the
Postgres type already exists). Both had to be added by hand in the
migration file - autogenerate output should always be read, not applied
blindly, and this is a concrete example of where it falls short.

## Running it

### Docker Compose (recommended)

```bash
docker compose up -d --build
```

The app image is built from `Dockerfile.uv` (uv, following the official uv
Docker guide). `Dockerfile.pip` is an equivalent pip-based version - to use
it instead, change the `dockerfile:` line under `app: build:` in
`compose.yaml`.

This builds the app image and starts Postgres, MongoDB and the API together —
the app waits for both databases to report healthy before starting. Data
persists in named volumes (`pgdata`, `mongodata`) across restarts.

```bash
curl http://localhost:8000/health
```

To stop:

```bash
docker compose down        # keeps data
docker compose down -v     # also wipes the database volumes
```

### Running the app locally (without Docker)

Requires a Postgres instance reachable at the `DATABASE_URL` in `.env`
(e.g. just run `docker compose up -d db mongo` for the databases only).
MongoDB is optional here: without it the app works, and only the traces are
skipped.
Dependencies are managed with [uv](https://docs.astral.sh/uv/) - install
it once, then:

```bash
uv sync                                  # create/update .venv from uv.lock
uv run alembic upgrade head              # apply migrations
uv run uvicorn app.main:app --reload
```

### Configuration

Copy `.env.example` to `.env` and fill in:

| Variable                    | Purpose                                                                | Default (local)                                                   |
|-----------------------------|------------------------------------------------------------------------|-------------------------------------------------------------------|
| `GROQ_API_KEY`              | Groq API key (free tier at console.groq.com)                           | —                                                                 |
| `GROQ_MODEL`                | Groq model to use                                                      | `openai/gpt-oss-120b`                                             |
| `DATABASE_URL`              | Postgres connection string (asyncpg driver)                            | `postgresql+asyncpg://root:root@localhost:5432/incident_analyzer` |
| `DEDUP_WINDOW_MINUTES`      | How long a repeat error is treated as a duplicate                      | `10`                                                              |
| `LLM_MAX_ATTEMPTS`          | Max attempts before falling back (see Retry & fallback)                | `3`                                                               |
| `LLM_RETRY_BACKOFF_SECONDS` | Backoff multiplier between retries on transient API errors             | `0.5`                                                             |
| `TASK_DECOMPOSITION`        | `true`: classify + prioritize as two calls; `false`: one combined call | `false`                                                           |
| `MONGO_URL`                 | MongoDB connection string (LLM call traces)                            | `mongodb://root:root@localhost:27017/`                            |
| `MONGO_DB`                  | MongoDB database name                                                  | `incident_analyzer`                                               |
| `LLM_TRACE_TTL_DAYS`        | How long LLM call traces are kept before MongoDB deletes them          | `30`                                                              |

Inside `compose.yaml`, `DATABASE_URL` and `MONGO_URL` are overridden to
point at the `db` and `mongo` service hostnames instead of `localhost`, since
that's how containers reach each other on the compose network.

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

### `GET /stats/llm`

Statistics over the LLM call traces (see *Statistics* above).

| Query param | Meaning                                         | Default |
|-------------|-------------------------------------------------|---------|
| `days`      | how many days back to look (1-365)              | `7`     |
| `source`    | `api` (real traffic), `eval`, or `all`          | `api`   |

```json
{
  "since": "2026-11-17T20:50:35Z",
  "source": "api",
  "overview": {
    "runs": 4, "fallbacks": 0, "fallback_rate": 0.0, "retries": 0,
    "latency": {"mean_ms": 1090.8, "p50_ms": 954.0, "p95_ms": 1675.0, "max_ms": 1675}
  },
  "by_mode_model": [
    {"mode": "decomposed", "model": "openai/gpt-oss-120b", "runs": 1, "fallback_rate": 0.0,
     "mean_retry_count": 0.0, "latency": {"mean_ms": 1675.0, "p50_ms": 1675.0, "p95_ms": 1675.0, "max_ms": 1675}},
    {"mode": "single", "model": "openai/gpt-oss-120b", "runs": 3, "...": "..."}
  ],
  "steps": [
    {"step": "analyze", "attempts": 3, "failed": 0, "failure_rate": 0.0, "statuses": {"ok": 3}},
    {"step": "classify", "attempts": 1, "failed": 0, "failure_rate": 0.0, "statuses": {"ok": 1}},
    {"step": "prioritize", "attempts": 1, "failed": 0, "failure_rate": 0.0, "statuses": {"ok": 1}}
  ]
}
```

`overview` is `null` when no trace matches. If MongoDB is unreachable, the
endpoint returns `503`. Unlike saving a trace, statistics can't be best
effort, because without the store there is nothing to report. The driver's
error (internal host names, timeouts) is logged, not returned to the client.

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
