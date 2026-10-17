"""
Tests for models/_retry.py — the shared transient-error retry helper added
after real Groq/Llama runs showed a large fraction of RateLimitError and
APIConnectionError failures that a bare single-attempt call could not
distinguish from genuine model errors on disk.
"""
from __future__ import annotations
import unittest
from unittest.mock import patch

from models._retry import call_with_retry


class FakeRateLimitError(Exception):
    pass


class FakeAuthError(Exception):
    pass


# Patch the module's retryable-name set indirectly by naming our fake
# exception classes to match real provider exception names, since
# call_with_retry matches by type(e).__name__, not by import identity.
FakeRateLimitError.__name__ = "RateLimitError"
FakeAuthError.__name__ = "AuthenticationError"


class TestCallWithRetry(unittest.TestCase):
    def test_succeeds_immediately_with_no_retries(self):
        calls = []

        def fn():
            calls.append(1)
            return "ok"

        result, attempts = call_with_retry(fn)
        self.assertEqual(result, "ok")
        self.assertEqual(attempts, 1)
        self.assertEqual(len(calls), 1)

    def test_retries_on_retryable_error_then_succeeds(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise FakeRateLimitError("rate limited")
            return "recovered"

        with patch("models._retry.time.sleep", return_value=None):  # skip real backoff delay in tests
            result, attempts = call_with_retry(fn, base_delay_s=0.01, max_delay_s=0.02)

        self.assertEqual(result, "recovered")
        self.assertEqual(attempts, 3)
        self.assertEqual(calls["n"], 3)

    def test_raises_immediately_on_non_retryable_error(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise FakeAuthError("bad key")

        with self.assertRaises(FakeAuthError):
            call_with_retry(fn, max_attempts=5)
        self.assertEqual(calls["n"], 1, "non-retryable errors must not be retried")

    def test_gives_up_after_max_attempts(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise FakeRateLimitError("still limited")

        with patch("models._retry.time.sleep", return_value=None):
            with self.assertRaises(FakeRateLimitError):
                call_with_retry(fn, max_attempts=3, base_delay_s=0.01, max_delay_s=0.02)
        self.assertEqual(calls["n"], 3, "must stop after exactly max_attempts tries")


if __name__ == "__main__":
    unittest.main()
