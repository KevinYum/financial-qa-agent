"""News search tool — fetches financial news via Brave Search API."""

import httpx

from ..config import settings

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


async def fetch_news(question: str, parse_result: dict | None = None) -> str:
    """Search for recent financial news related to the question.

    Uses parse_result.news_query (refined by parse LLM) if available,
    otherwise falls back to the raw question. Returns formatted results.
    """
    if not settings.brave_api_key:
        return "News search is not configured (missing BRAVE_API_KEY)."

    parse_result = parse_result or {}
    query = parse_result.get("news_query") or question

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": settings.brave_api_key,
    }
    params = {
        "q": query,
        "count": 5,
        "freshness": "pw",  # past week
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            BRAVE_SEARCH_URL, headers=headers, params=params
        )
        response.raise_for_status()
        data = response.json()

    results = data.get("web", {}).get("results", [])
    if not results:
        return "No recent news found for this query."

    parts: list[str] = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "No title")
        description = r.get("description", "No description")
        url = r.get("url", "")
        age = r.get("age", "")
        parts.append(
            f"[{i}] {title}\n    {description}\n    Source: {url}\n    Age: {age}"
        )

    return "\n\n".join(parts)
