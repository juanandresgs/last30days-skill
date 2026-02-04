"""Tests for HTTP retry delay logic.

@decision Tests target _get_retry_delay directly rather than mocking time.sleep
in the full request() path — the helper is a pure function, making tests fast
and deterministic. Integration-level retry behavior is validated by live runs.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from lib import http


class TestGetRetryDelay(unittest.TestCase):
    """Tests for _get_retry_delay helper."""

    @patch("lib.http.random.random", return_value=0.0)
    def test_exponential_growth(self, _mock_random):
        """Non-429 delays grow exponentially from RETRY_BASE_DELAY."""
        d0 = http._get_retry_delay(attempt=0, is_rate_limit=False)
        d1 = http._get_retry_delay(attempt=1, is_rate_limit=False)
        d2 = http._get_retry_delay(attempt=2, is_rate_limit=False)

        self.assertAlmostEqual(d0, http.RETRY_BASE_DELAY)       # 2.0
        self.assertAlmostEqual(d1, http.RETRY_BASE_DELAY * 2)   # 4.0
        self.assertAlmostEqual(d2, http.RETRY_BASE_DELAY * 4)   # 8.0

    @patch("lib.http.random.random", return_value=0.0)
    def test_rate_limit_uses_higher_base(self, _mock_random):
        """429 errors use RETRY_429_BASE_DELAY (higher than normal)."""
        d0 = http._get_retry_delay(attempt=0, is_rate_limit=True)
        d1 = http._get_retry_delay(attempt=1, is_rate_limit=True)

        self.assertAlmostEqual(d0, http.RETRY_429_BASE_DELAY)       # 5.0
        self.assertAlmostEqual(d1, http.RETRY_429_BASE_DELAY * 2)   # 10.0
        self.assertGreater(d0, http.RETRY_BASE_DELAY)

    def test_retry_after_honored(self):
        """Retry-After header value is used directly."""
        delay = http._get_retry_delay(attempt=0, is_rate_limit=True, retry_after=30.0)
        self.assertAlmostEqual(delay, 30.0)

    def test_retry_after_capped(self):
        """Retry-After is capped at RETRY_MAX_DELAY."""
        delay = http._get_retry_delay(attempt=0, retry_after=120.0)
        self.assertAlmostEqual(delay, http.RETRY_MAX_DELAY)

    @patch("lib.http.random.random", return_value=0.0)
    def test_exponential_capped_at_max(self, _mock_random):
        """Exponential backoff never exceeds RETRY_MAX_DELAY."""
        delay = http._get_retry_delay(attempt=10, is_rate_limit=True)
        self.assertLessEqual(delay, http.RETRY_MAX_DELAY)

    @patch("lib.http.random.random", return_value=1.0)
    def test_jitter_applied(self, _mock_random):
        """Jitter adds up to 25% on top of base delay."""
        delay = http._get_retry_delay(attempt=0, is_rate_limit=False)
        base = http.RETRY_BASE_DELAY
        # With random=1.0, jitter = base * 0.25 * 1.0
        self.assertAlmostEqual(delay, base + base * 0.25)

    def test_jitter_is_non_negative(self):
        """Delay is always >= the base exponential value."""
        for attempt in range(5):
            delay = http._get_retry_delay(attempt, is_rate_limit=False)
            min_delay = min(http.RETRY_BASE_DELAY * (2 ** attempt), http.RETRY_MAX_DELAY)
            self.assertGreaterEqual(delay, min_delay)


class TestHTTPErrorRetryAfter(unittest.TestCase):
    """Tests for HTTPError retry_after field."""

    def test_retry_after_stored(self):
        err = http.HTTPError("rate limited", status_code=429, retry_after=15.0)
        self.assertEqual(err.retry_after, 15.0)

    def test_retry_after_defaults_none(self):
        err = http.HTTPError("server error", status_code=500)
        self.assertIsNone(err.retry_after)


class TestConstants(unittest.TestCase):
    """Verify backoff constants are sensible."""

    def test_429_base_higher_than_normal(self):
        self.assertGreater(http.RETRY_429_BASE_DELAY, http.RETRY_BASE_DELAY)

    def test_max_delay_caps_reasonable(self):
        self.assertGreaterEqual(http.RETRY_MAX_DELAY, 30.0)
        self.assertLessEqual(http.RETRY_MAX_DELAY, 120.0)


if __name__ == "__main__":
    unittest.main()
