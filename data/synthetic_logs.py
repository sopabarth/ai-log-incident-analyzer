"""
Synthetic log / error examples used for local testing and for the eval set.

Each example pairs a raw, messy log entry (the kind of thing you'd actually
scrape out of stdout/stderr or a log aggregator) with a hand-assigned ground
truth label. The labels are assigned by us, not by the model, so eval/run_eval.py
has a trustworthy baseline to score the pipeline's output against.

Deliberately kept free of any `app.*` imports (plain dataclass + str fields)
so this module can be used standalone - to seed a DB, dump JSON, or run the
eval script - without pulling in FastAPI/Groq as a dependency.

Categories mirror app.schemas.ErrorCategory:
  database_timeout, null_pointer, rate_limit_exceeded, auth_failure,
  network_partial_failure, validation_error, unknown
Priorities mirror app.schemas.Priority: critical, high, medium, low
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
from pathlib import Path


@dataclass
class SyntheticLogExample:
    service: str
    environment: str  # prod | staging | dev
    timestamp: str  # ISO 8601
    raw_text: str
    expected_category: str
    expected_priority: str
    expected_needs_human_review: bool
    notes: str = ""


_BASE_TIME = datetime(2026, 9, 10, 12, 0, 0)


def _ts(offset_minutes: int) -> str:
    """Spread timestamps out so a batch looks like a real incident window."""
    return (_BASE_TIME + timedelta(minutes=offset_minutes)).isoformat()


SYNTHETIC_LOGS: list[SyntheticLogExample] = [
    # ============== DATABASE_TIMEOUT ==============
    SyntheticLogExample(
        service="payments-api",
        environment="prod",
        timestamp=_ts(0),
        raw_text=(
            'Traceback (most recent call last):\n'
            '  File "app/db/session.py", line 42, in get_connection\n'
            '    conn = pool.acquire(timeout=5)\n'
            '  File "app/services/payment.py", line 88, in charge_card\n'
            '    with db.session() as session:\n'
            'psycopg2.OperationalError: timeout expired\n'
        ),
        expected_category="database_timeout",
        expected_priority="critical",
        expected_needs_human_review=False,
        notes="Prod payments path blocked on DB timeout - revenue impacting.",
    ),
    SyntheticLogExample(
        service="orders-api",
        environment="prod",
        timestamp=_ts(1),
        raw_text=(
            "2026-09-10T12:01:04Z ERROR orders-api db_pool: "
            "could not acquire connection from pool within 5000ms, "
            "active=50 idle=0 waiting=127"
        ),
        expected_category="database_timeout",
        expected_priority="critical",
        expected_needs_human_review=False,
        notes="Pool fully exhausted, large queue - systemic outage.",
    ),
    SyntheticLogExample(
        service="reporting-service",
        environment="prod",
        timestamp=_ts(2),
        raw_text=(
            "com.reporting.db.QueryTimeoutException: Statement cancelled "
            "because timeout=30000ms exceeded, query=nightly_revenue_rollup\n"
            "\tat com.reporting.jobs.NightlyRollup.run(NightlyRollup.java:112)\n"
            "\tat java.base/java.lang.Thread.run(Thread.java:833)"
        ),
        expected_category="database_timeout",
        expected_priority="medium",
        expected_needs_human_review=False,
        notes="Prod but a background batch job, not user-facing.",
    ),
    SyntheticLogExample(
        service="inventory-service",
        environment="staging",
        timestamp=_ts(3),
        raw_text=(
            "pq: canceling statement due to statement timeout\n"
            "  at inventory.(*Repo).UpdateStock (repo.go:77)\n"
            "  at inventory.(*Handler).Reserve (handler.go:31)"
        ),
        expected_category="database_timeout",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Same failure shape as prod ones, but staging - lower urgency.",
    ),
    SyntheticLogExample(
        service="billing-service",
        environment="prod",
        timestamp=_ts(4),
        raw_text=(
            "MongoTimeoutError: Server selection timed out after 5000 ms, "
            "topology description: ReplicaSetNoPrimary\n"
            "    at Timeout._onTimeout (mongodb/lib/sdam/topology.js:298)"
        ),
        expected_category="database_timeout",
        expected_priority="critical",
        expected_needs_human_review=False,
        notes="No primary available - full replica set outage.",
    ),
    SyntheticLogExample(
        service="user-service",
        environment="dev",
        timestamp=_ts(5),
        raw_text=(
            "sqlalchemy.exc.OperationalError: (mysql) Lock wait timeout "
            "exceeded; try restarting transaction"
        ),
        expected_category="database_timeout",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Dev environment, likely a local migration/seed script.",
    ),

    # ============== NULL_POINTER ==============
    SyntheticLogExample(
        service="checkout-worker",
        environment="prod",
        timestamp=_ts(6),
        raw_text=(
            "Exception in thread \"main\" java.lang.NullPointerException: "
            "Cannot invoke \"Cart.getItems()\" because \"cart\" is null\n"
            "\tat com.shop.checkout.CartService.finalize(CartService.java:64)\n"
            "\tat com.shop.checkout.CheckoutWorker.process(CheckoutWorker.java:29)"
        ),
        expected_category="null_pointer",
        expected_priority="critical",
        expected_needs_human_review=False,
        notes="Worker crash on the checkout path in prod.",
    ),
    SyntheticLogExample(
        service="recommendation-engine",
        environment="prod",
        timestamp=_ts(7),
        raw_text=(
            'Traceback (most recent call last):\n'
            '  File "app/rank.py", line 51, in score_items\n'
            '    weight = user.profile.preferences["category"]\n'
            "AttributeError: 'NoneType' object has no attribute 'preferences'\n"
        ),
        expected_category="null_pointer",
        expected_priority="medium",
        expected_needs_human_review=False,
        notes="Degrades personalization but the page still renders (fallback ranking).",
    ),
    SyntheticLogExample(
        service="search-api",
        environment="prod",
        timestamp=_ts(8),
        raw_text=(
            "panic: runtime error: invalid memory address or nil pointer dereference\n"
            "[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x4a5c31]\n"
            "goroutine 42 [running]:\n"
            "main.(*Index).Lookup(0x0, {0xc0001a4000, 0x8})\n"
            "\t/app/search/index.go:203 +0x1b"
        ),
        expected_category="null_pointer",
        expected_priority="critical",
        expected_needs_human_review=False,
        notes="Goroutine panic crashing the process - search fully down.",
    ),
    SyntheticLogExample(
        service="notifications-service",
        environment="staging",
        timestamp=_ts(9),
        raw_text=(
            "TypeError: Cannot read properties of undefined (reading 'email')\n"
            "    at buildPayload (src/notify/template.js:18:22)\n"
            "    at sendWelcomeEmail (src/notify/index.js:9:15)"
        ),
        expected_category="null_pointer",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Staging, non-critical notification path.",
    ),
    SyntheticLogExample(
        service="orders-api",
        environment="dev",
        timestamp=_ts(10),
        raw_text=(
            "kotlin.KotlinNullPointerException\n"
            "\tat com.shop.orders.OrderMapper.toDto(OrderMapper.kt:22)\n"
            "\tat com.shop.orders.OrderController.get(OrderController.kt:40)"
        ),
        expected_category="null_pointer",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Local dev, likely missing test fixture data.",
    ),
    SyntheticLogExample(
        service="billing-service",
        environment="prod",
        timestamp=_ts(11),
        raw_text=(
            'Traceback (most recent call last):\n'
            '  File "app/invoices.py", line 30, in apply_discount\n'
            '    total = order.discount.percent * order.subtotal\n'
            "AttributeError: 'NoneType' object has no attribute 'percent'\n"
        ),
        expected_category="null_pointer",
        expected_priority="high",
        expected_needs_human_review=False,
        notes="Prod invoice generation broken for orders with no discount object.",
    ),

    # ============== RATE_LIMIT_EXCEEDED ==============
    SyntheticLogExample(
        service="payments-api",
        environment="prod",
        timestamp=_ts(12),
        raw_text=(
            "2026-09-10T12:12:00Z ERROR payments-api stripe_client: "
            "request failed status=429 body={\"error\":\"rate_limit\","
            "\"message\":\"Too many requests hit the API too quickly\"}"
        ),
        expected_category="rate_limit_exceeded",
        expected_priority="high",
        expected_needs_human_review=False,
        notes="External payment provider throttling - blocks checkouts.",
    ),
    SyntheticLogExample(
        service="search-api",
        environment="prod",
        timestamp=_ts(13),
        raw_text=(
            "botocore.exceptions.ClientError: An error occurred "
            "(ThrottlingException) when calling the Query operation: "
            "Rate exceeded"
        ),
        expected_category="rate_limit_exceeded",
        expected_priority="medium",
        expected_needs_human_review=False,
        notes="DynamoDB throttling under a traffic spike, has retry/backoff.",
    ),
    SyntheticLogExample(
        service="email-worker",
        environment="prod",
        timestamp=_ts(14),
        raw_text=(
            "429 Too Many Requests from sendgrid.com: "
            "\"Maximum credits exceeded, upgrade plan or wait for reset\""
        ),
        expected_category="rate_limit_exceeded",
        expected_priority="medium",
        expected_needs_human_review=False,
        notes="Emails queue and retry later, not user-blocking in real time.",
    ),
    SyntheticLogExample(
        service="user-service",
        environment="prod",
        timestamp=_ts(15),
        raw_text=(
            "WARN rate_limiter: client_id=mobile-app blocked, "
            "429 responses sustained for 3m, threshold=100req/s"
        ),
        expected_category="rate_limit_exceeded",
        expected_priority="high",
        expected_needs_human_review=False,
        notes="Own rate limiter blocking the primary mobile client for 3 minutes.",
    ),
    SyntheticLogExample(
        service="inventory-service",
        environment="staging",
        timestamp=_ts(16),
        raw_text=(
            "redis.exceptions.ResponseError: rate limit exceeded for "
            "key stock_sync:staging, retry_after=12s"
        ),
        expected_category="rate_limit_exceeded",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Staging sync job, self-recovers.",
    ),
    SyntheticLogExample(
        service="reporting-service",
        environment="dev",
        timestamp=_ts(17),
        raw_text=(
            "requests.exceptions.HTTPError: 429 Client Error: "
            "Too Many Requests for url: https://api.analytics.dev/export"
        ),
        expected_category="rate_limit_exceeded",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Dev script hammering a sandbox API.",
    ),

    # ============== AUTH_FAILURE ==============
    SyntheticLogExample(
        service="auth-gateway",
        environment="prod",
        timestamp=_ts(18),
        raw_text=(
            "2026-09-10T12:18:00Z ERROR auth-gateway jwt: "
            "token validation failed for all incoming requests since 12:15 - "
            "signing key rotation mismatch, kid=2026-09-a not found in JWKS cache"
        ),
        expected_category="auth_failure",
        expected_priority="critical",
        expected_needs_human_review=False,
        notes="Every request is being rejected - total outage across all services.",
    ),
    SyntheticLogExample(
        service="user-service",
        environment="prod",
        timestamp=_ts(19),
        raw_text=(
            "401 Unauthorized: invalid credentials for user_id=88213, "
            "attempt=4, account temporarily locked"
        ),
        expected_category="auth_failure",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Single user, normal login-failure/lockout flow.",
    ),
    SyntheticLogExample(
        service="orders-api",
        environment="prod",
        timestamp=_ts(20),
        raw_text=(
            "oauth2.errors.InvalidGrantError: refresh token expired for "
            "service_account=orders-to-warehouse-sync, cannot refresh access token"
        ),
        expected_category="auth_failure",
        expected_priority="high",
        expected_needs_human_review=False,
        notes="Service-to-service integration broken - warehouse sync stalled.",
    ),
    SyntheticLogExample(
        service="notifications-service",
        environment="staging",
        timestamp=_ts(21),
        raw_text=(
            "LDAPBindError: Invalid credentials (staging-service-account), "
            "bind failed against ldap://staging-directory:389"
        ),
        expected_category="auth_failure",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Staging service account misconfig.",
    ),
    SyntheticLogExample(
        service="billing-service",
        environment="prod",
        timestamp=_ts(22),
        raw_text=(
            "403 Forbidden: API key 'live_sk_***7f2a' was revoked on "
            "2026-09-09, all requests to /v1/charges rejected"
        ),
        expected_category="auth_failure",
        expected_priority="critical",
        expected_needs_human_review=False,
        notes="Billing entirely unable to process charges.",
    ),
    SyntheticLogExample(
        service="search-api",
        environment="dev",
        timestamp=_ts(23),
        raw_text=(
            "AuthenticationError: no API key provided (did you forget to "
            "set SEARCH_API_KEY in .env.local?)"
        ),
        expected_category="auth_failure",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Local dev misconfiguration.",
    ),

    # ============== NETWORK_PARTIAL_FAILURE ==============
    SyntheticLogExample(
        service="checkout-worker",
        environment="prod",
        timestamp=_ts(24),
        raw_text=(
            "2026-09-10T12:24:00Z WARN checkout-worker http_client: "
            "connection reset by peer calling shipping-api.internal, "
            "retry 1/3 succeeded after 340ms"
        ),
        expected_category="network_partial_failure",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Transient blip, retry succeeded - no user impact.",
    ),
    SyntheticLogExample(
        service="orders-api",
        environment="prod",
        timestamp=_ts(25),
        raw_text=(
            "grpc: rpc error: code = Unavailable desc = connection error: "
            "desc = \"transport: Error while dialing dial tcp "
            "10.2.4.18:9090: i/o timeout\", target=inventory-service, "
            "affected_requests=~15% over last 5 min"
        ),
        expected_category="network_partial_failure",
        expected_priority="high",
        expected_needs_human_review=False,
        notes="Sustained partial failure rate against a core dependency.",
    ),
    SyntheticLogExample(
        service="user-service",
        environment="prod",
        timestamp=_ts(26),
        raw_text=(
            "ERROR dns_resolver: intermittent NXDOMAIN for "
            "auth-gateway.internal, ~8% of lookups failing, "
            "falling back to cached IP"
        ),
        expected_category="network_partial_failure",
        expected_priority="medium",
        expected_needs_human_review=False,
        notes="Degraded but mitigated via DNS cache fallback.",
    ),
    SyntheticLogExample(
        service="reporting-service",
        environment="staging",
        timestamp=_ts(27),
        raw_text=(
            "requests.exceptions.ConnectionError: HTTPSConnectionPool"
            "(host='staging-warehouse.internal', port=443): "
            "Max retries exceeded, Connection reset by peer"
        ),
        expected_category="network_partial_failure",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Staging warehouse connectivity blip.",
    ),
    SyntheticLogExample(
        service="payments-api",
        environment="prod",
        timestamp=_ts(28),
        raw_text=(
            "SSLError: [SSL: HANDSHAKE_FAILURE] TLS handshake failed "
            "connecting to fraud-check.internal, cert appears expired"
        ),
        expected_category="network_partial_failure",
        expected_priority="critical",
        expected_needs_human_review=False,
        notes="Expired cert blocks all calls to the fraud-check dependency on the payment path.",
    ),
    SyntheticLogExample(
        service="notifications-service",
        environment="dev",
        timestamp=_ts(29),
        raw_text=(
            "socket.timeout: connection to push-gateway.dev timed out "
            "after 2000ms, no retry configured"
        ),
        expected_category="network_partial_failure",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Dev push gateway not always up.",
    ),

    # ============== VALIDATION_ERROR ==============
    SyntheticLogExample(
        service="orders-api",
        environment="prod",
        timestamp=_ts(30),
        raw_text=(
            "pydantic.error_wrappers.ValidationError: 1 validation error "
            "for CreateOrderRequest\nquantity\n  ensure this value is "
            "greater than 0 (type=value_error.number.not_gt; limit_value=0)"
        ),
        expected_category="validation_error",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Bad client input, handled as a 4xx - not a system fault.",
    ),
    SyntheticLogExample(
        service="user-service",
        environment="prod",
        timestamp=_ts(31),
        raw_text=(
            "400 Bad Request: JSON body missing required field 'email' "
            "at /v1/users/register, malformed_requests=340 in last 10 min"
        ),
        expected_category="validation_error",
        expected_priority="medium",
        expected_needs_human_review=False,
        notes="High volume suggests an upstream client (mobile app release?) sending a bad payload.",
    ),
    SyntheticLogExample(
        service="billing-service",
        environment="prod",
        timestamp=_ts(32),
        raw_text=(
            "jakarta.validation.ConstraintViolationException: "
            "charge.amountCents: must be greater than or equal to 1"
        ),
        expected_category="validation_error",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Rejected at the boundary as designed.",
    ),
    SyntheticLogExample(
        service="inventory-service",
        environment="staging",
        timestamp=_ts(33),
        raw_text=(
            "json.decoder.JSONDecodeError: Expecting ',' delimiter: "
            "line 1 column 42 (char 41), payload truncated at 41 bytes"
        ),
        expected_category="validation_error",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Malformed JSON body from a staging test client.",
    ),
    SyntheticLogExample(
        service="search-api",
        environment="prod",
        timestamp=_ts(34),
        raw_text=(
            "ValueError: invalid enum value 'sort_by=popularityy' - "
            "expected one of ['relevance', 'price', 'popularity']"
        ),
        expected_category="validation_error",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Typo'd query param, handled gracefully.",
    ),
    SyntheticLogExample(
        service="orders-api",
        environment="dev",
        timestamp=_ts(35),
        raw_text=(
            "marshmallow.exceptions.ValidationError: {'shipping_address': "
            "['Missing data for required field.']}"
        ),
        expected_category="validation_error",
        expected_priority="low",
        expected_needs_human_review=False,
        notes="Dev/test data missing a field.",
    ),

    # ============== UNKNOWN / AMBIGUOUS (should trigger human review) ==============
    SyntheticLogExample(
        service="checkout-worker",
        environment="prod",
        timestamp=_ts(36),
        raw_text="ERROR: something went wrong. code=7 retrying...",
        expected_category="unknown",
        expected_priority="medium",
        expected_needs_human_review=True,
        notes="Too vague to classify confidently - no exception type or context.",
    ),
    SyntheticLogExample(
        service="payments-api",
        environment="prod",
        timestamp=_ts(37),
        raw_text=(
            "kernel: [123456.789] Out of memory: Killed process 4821 "
            "(python3) total-vm:4129852kB, anon-rss:3980112kB"
        ),
        expected_category="unknown",
        expected_priority="high",
        expected_needs_human_review=True,
        notes="OOM kill - real and severe, but not one of the known app-level categories.",
    ),
    SyntheticLogExample(
        service="orders-api",
        environment="prod",
        timestamp=_ts(38),
        raw_text=(
            "\\x00\\x00garbled\\xffoutput\\x1b[31mSEG\\x1b[0mFAULT??"
            "core dumped @ 0x7fff"
        ),
        expected_category="unknown",
        expected_priority="medium",
        expected_needs_human_review=True,
        notes="Corrupted/binary log output, unparseable.",
    ),
    SyntheticLogExample(
        service="recommendation-engine",
        environment="prod",
        timestamp=_ts(39),
        raw_text=(
            "custom_metric_alert: model_drift_score=0.83 exceeded "
            "threshold=0.7, no exception raised, model still serving"
        ),
        expected_category="unknown",
        expected_priority="medium",
        expected_needs_human_review=True,
        notes="A monitoring alert, not a traditional error/exception - new pattern for the classifier.",
    ),
    SyntheticLogExample(
        service="inventory-service",
        environment="staging",
        timestamp=_ts(40),
        raw_text="WARN: retrying operation (attempt 2)",
        expected_category="unknown",
        expected_priority="low",
        expected_needs_human_review=True,
        notes="No indication of what operation or why it failed.",
    ),
]


def to_dicts() -> list[dict]:
    return [asdict(example) for example in SYNTHETIC_LOGS]


if __name__ == "__main__":
    # Dump to data/synthetic_logs.json so the eval script (and anything else
    # that shouldn't import Python) can consume it without touching this module.
    out_path = Path(__file__).parent / "synthetic_logs.json"
    out_path.write_text(json.dumps(to_dicts(), indent=2), encoding="utf-8")
    print(f"Wrote {len(SYNTHETIC_LOGS)} synthetic examples to {out_path}")
