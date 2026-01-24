"""WebSearch module for last30days skill.

NOTE: WebSearch uses Claude's built-in WebSearch tool, which runs INSIDE Claude Code.
Unlike Reddit/X which use external APIs, WebSearch results are obtained by Claude
directly and passed to this module for normalization and scoring.

The typical flow is:
1. Claude invokes WebSearch tool with the topic
2. Claude passes results to parse_websearch_results()
3. Results are normalized into WebSearchItem objects
"""

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from . import schema


# Domains to exclude (Reddit and X are handled separately)
EXCLUDED_DOMAINS = {
    "reddit.com",
    "www.reddit.com",
    "old.reddit.com",
    "twitter.com",
    "www.twitter.com",
    "x.com",
    "www.x.com",
    "mobile.twitter.com",
}


def extract_domain(url: str) -> str:
    """Extract the domain from a URL.

    Args:
        url: Full URL

    Returns:
        Domain string (e.g., "medium.com")
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        # Remove www. prefix for cleaner display
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return ""


def is_excluded_domain(url: str) -> bool:
    """Check if URL is from an excluded domain (Reddit/X).

    Args:
        url: URL to check

    Returns:
        True if URL should be excluded
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return domain in EXCLUDED_DOMAINS
    except Exception:
        return False


def parse_websearch_results(
    results: List[Dict[str, Any]],
    topic: str,
) -> List[Dict[str, Any]]:
    """Parse WebSearch results into normalized format.

    This function expects results from Claude's WebSearch tool.
    Each result should have: title, url, snippet, and optionally date/relevance.

    Args:
        results: List of WebSearch result dicts
        topic: Original search topic (for context)

    Returns:
        List of normalized item dicts ready for WebSearchItem creation
    """
    items = []

    for i, result in enumerate(results):
        if not isinstance(result, dict):
            continue

        url = result.get("url", "")
        if not url:
            continue

        # Skip Reddit/X URLs (handled separately)
        if is_excluded_domain(url):
            continue

        title = str(result.get("title", "")).strip()
        snippet = str(result.get("snippet", result.get("description", ""))).strip()

        if not title and not snippet:
            continue

        # Parse date if provided
        date = result.get("date")
        date_confidence = "low"
        if date:
            # Validate date format
            if re.match(r'^\d{4}-\d{2}-\d{2}$', str(date)):
                date_confidence = "med"  # WebSearch dates are often approximate
            else:
                date = None

        # Get relevance if provided, default to 0.5
        relevance = result.get("relevance", 0.5)
        try:
            relevance = min(1.0, max(0.0, float(relevance)))
        except (TypeError, ValueError):
            relevance = 0.5

        item = {
            "id": f"W{i+1}",
            "title": title[:200],  # Truncate long titles
            "url": url,
            "source_domain": extract_domain(url),
            "snippet": snippet[:500],  # Truncate long snippets
            "date": date,
            "date_confidence": date_confidence,
            "relevance": relevance,
            "why_relevant": str(result.get("why_relevant", "")).strip(),
        }

        items.append(item)

    return items


def normalize_websearch_items(
    items: List[Dict[str, Any]],
    from_date: str,
    to_date: str,
) -> List[schema.WebSearchItem]:
    """Convert parsed dicts to WebSearchItem objects.

    Args:
        items: List of parsed item dicts
        from_date: Start of date range (YYYY-MM-DD)
        to_date: End of date range (YYYY-MM-DD)

    Returns:
        List of WebSearchItem objects
    """
    result = []

    for item in items:
        web_item = schema.WebSearchItem(
            id=item["id"],
            title=item["title"],
            url=item["url"],
            source_domain=item["source_domain"],
            snippet=item["snippet"],
            date=item.get("date"),
            date_confidence=item.get("date_confidence", "low"),
            relevance=item.get("relevance", 0.5),
            why_relevant=item.get("why_relevant", ""),
        )
        result.append(web_item)

    return result


def dedupe_websearch(items: List[schema.WebSearchItem]) -> List[schema.WebSearchItem]:
    """Remove duplicate WebSearch items.

    Deduplication is based on URL.

    Args:
        items: List of WebSearchItem objects

    Returns:
        Deduplicated list
    """
    seen_urls = set()
    result = []

    for item in items:
        # Normalize URL for comparison
        url_key = item.url.lower().rstrip("/")
        if url_key not in seen_urls:
            seen_urls.add(url_key)
            result.append(item)

    return result
