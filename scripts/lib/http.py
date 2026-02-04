"""HTTP utilities for last30days skill (stdlib only)."""

import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional
from urllib.parse import urlencode

DEFAULT_TIMEOUT = 30
DEBUG = os.environ.get("LAST30DAYS_DEBUG", "").lower() in ("1", "true", "yes")


def log(msg: str):
    """Log debug message to stderr."""
    if DEBUG:
        sys.stderr.write(f"[DEBUG] {msg}\n")
        sys.stderr.flush()
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
RETRY_429_BASE_DELAY = 5.0
RETRY_MAX_DELAY = 60.0
USER_AGENT = "last30days-skill/1.0 (Claude Code Skill)"


class HTTPError(Exception):
    """HTTP request error with status code."""
    def __init__(self, message: str, status_code: Optional[int] = None,
                 body: Optional[str] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.retry_after = retry_after


def _get_retry_delay(attempt: int, is_rate_limit: bool = False,
                     retry_after: Optional[float] = None) -> float:
    """Calculate retry delay with exponential backoff and jitter.

    @decision Exponential backoff with separate 429 base — Reddit and X APIs
    rate-limit aggressively; linear 1-3s was insufficient. 429s use 5s base
    (vs 2s for other errors) because rate-limit windows are typically 60s.
    Retry-After honored when present, capped at RETRY_MAX_DELAY.

    Args:
        attempt: Zero-based attempt number (0 = first retry)
        is_rate_limit: True if the error was a 429
        retry_after: Value from Retry-After header, if present

    Returns:
        Delay in seconds
    """
    if retry_after is not None:
        delay = min(retry_after, RETRY_MAX_DELAY)
        return delay

    base = RETRY_429_BASE_DELAY if is_rate_limit else RETRY_BASE_DELAY
    delay = base * (2 ** attempt)
    delay = min(delay, RETRY_MAX_DELAY)
    # Add 25% jitter
    jitter = delay * 0.25 * random.random()
    return delay + jitter


def request(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """Make an HTTP request and return JSON response.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Request URL
        headers: Optional headers dict
        json_data: Optional JSON body (for POST)
        timeout: Request timeout in seconds
        retries: Number of retries on failure

    Returns:
        Parsed JSON response

    Raises:
        HTTPError: On request failure
    """
    headers = headers or {}
    headers.setdefault("User-Agent", USER_AGENT)

    data = None
    if json_data is not None:
        data = json.dumps(json_data).encode('utf-8')
        headers.setdefault("Content-Type", "application/json")

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    log(f"{method} {url}")
    if json_data:
        log(f"Payload keys: {list(json_data.keys())}")

    last_error = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = response.read().decode('utf-8')
                log(f"Response: {response.status} ({len(body)} bytes)")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            body = None
            try:
                body = e.read().decode('utf-8')
            except:
                pass
            log(f"HTTP Error {e.code}: {e.reason}")
            if body:
                log(f"Error body: {body[:500]}")

            # Parse Retry-After header if present
            retry_after = None
            retry_after_raw = e.headers.get("Retry-After") if e.headers else None
            if retry_after_raw:
                try:
                    retry_after = float(retry_after_raw)
                except (ValueError, TypeError):
                    pass

            last_error = HTTPError(f"HTTP {e.code}: {e.reason}", e.code, body, retry_after)

            # Don't retry client errors (4xx) except rate limits
            if 400 <= e.code < 500 and e.code != 429:
                raise last_error

            if attempt < retries - 1:
                is_rate_limit = (e.code == 429)
                delay = _get_retry_delay(attempt, is_rate_limit, retry_after)
                log(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
        except urllib.error.URLError as e:
            log(f"URL Error: {e.reason}")
            last_error = HTTPError(f"URL Error: {e.reason}")
            if attempt < retries - 1:
                delay = _get_retry_delay(attempt)
                log(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(delay)
        except json.JSONDecodeError as e:
            log(f"JSON decode error: {e}")
            last_error = HTTPError(f"Invalid JSON response: {e}")
            raise last_error
        except (OSError, TimeoutError, ConnectionResetError) as e:
            # Handle socket-level errors (connection reset, timeout, etc.)
            log(f"Connection error: {type(e).__name__}: {e}")
            last_error = HTTPError(f"Connection error: {type(e).__name__}: {e}")
            if attempt < retries - 1:
                delay = _get_retry_delay(attempt)
                log(f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{retries})")
                time.sleep(delay)

    if last_error:
        raise last_error
    raise HTTPError("Request failed with no error details")


def get(url: str, headers: Optional[Dict[str, str]] = None, **kwargs) -> Dict[str, Any]:
    """Make a GET request."""
    return request("GET", url, headers=headers, **kwargs)


def post(url: str, json_data: Dict[str, Any], headers: Optional[Dict[str, str]] = None, **kwargs) -> Dict[str, Any]:
    """Make a POST request with JSON body."""
    return request("POST", url, headers=headers, json_data=json_data, **kwargs)


def get_reddit_json(path: str) -> Dict[str, Any]:
    """Fetch Reddit thread JSON.

    Args:
        path: Reddit path (e.g., /r/subreddit/comments/id/title)

    Returns:
        Parsed JSON response
    """
    # Ensure path starts with /
    if not path.startswith('/'):
        path = '/' + path

    # Remove trailing slash and add .json
    path = path.rstrip('/')
    if not path.endswith('.json'):
        path = path + '.json'

    url = f"https://www.reddit.com{path}?raw_json=1"

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }

    return get(url, headers=headers)
