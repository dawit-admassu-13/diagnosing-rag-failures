"""
Shared transient-error retry helper for all three model adapters.

Rationale (added 2026-08-15): during the real Llama/Groq backfill run we
observed a large fraction of calls failing with RateLimitError and
APIConnectionError — ordinary transient provider-side issues, not a sign the
model/config is wrong. Without retries, a single-attempt adapter permanently
records these as an "error" result, indistinguishable on disk from a real
model refusal/limitation. Retrying a bounded number of times with backoff
before giving up keeps genuine model behavior (e.g. actual refusals,
malformed prompts) separated from ordinary network/rate-limit noise.

Applied identically to all three adapters (Claude, GPT-4.1, Llama) so no
model gets more retry attempts than another — retry behavior is
infrastructure robustness, not a generation-protocol parameter, but it must
still be applied uniformly for fairness and reproducibility.
"""
from __future__ import annotations
import random
import time

# Exception class names (by string, not import — avoids importing every
# provider SDK from this shared module) that are safe to retry: rate limits,
# transient connection drops, timeouts, and 5xx-class server errors. NOT
# retried: auth errors, bad-request/validation errors, permission errors —
# those indicate a real misconfiguration that retrying cannot fix.
RETRYABLE_EXCEPTION_NAMES = {
    "RateLimitError",
    "APIConnectionError",
    "APITimeoutError",
    "InternalServerError",
    "ServiceUnavailableError",
    "Timeout",
    "ConnectionError",
    "ConnectTimeout",
    "ReadTimeout",
}

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY_S = 2.0
DEFAULT_MAX_DELAY_S = 60.0


def call_with_retry(fn, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
                     base_delay_s: float = DEFAULT_BASE_DELAY_S,
                     max_delay_s: float = DEFAULT_MAX_DELAY_S):
    """
    Call fn() with no arguments. Retries with exponential backoff + jitter
    on transient errors (see RETRYABLE_EXCEPTION_NAMES). Returns
    (result, attempts_used) on success. Raises the final exception if all
    attempts are exhausted or the error is not retryable — the caller's
    existing except-Exception block turns that into a normal error record,
    so no adapter's error-handling contract changes.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = fn()
            return result, attempt
        except Exception as e:  # noqa: BLE001 — inspect, then re-raise or retry
            last_exc = e
            is_retryable = type(e).__name__ in RETRYABLE_EXCEPTION_NAMES
            if not is_retryable or attempt == max_attempts:
                raise
            delay = min(max_delay_s, base_delay_s * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)  # jitter to avoid thundering herd
            time.sleep(delay)
    raise last_exc  # pragma: no cover — unreachable, loop always returns or raises
