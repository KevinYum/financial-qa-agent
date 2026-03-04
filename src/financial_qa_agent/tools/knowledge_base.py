"""Knowledge base tool — ChromaDB local search and Brave web search.

Two public functions:
- ``fetch_local_knowledge`` — query ChromaDB only (semantic vector search)
- ``fetch_web_knowledge``  — query Brave web API, store results in ChromaDB

Embedding: ChromaDB's default embedding function (all-MiniLM-L6-v2 via
onnxruntime, 384-dim). The model is bundled with chromadb — no extra
download or API key required. Both ``query`` and ``upsert`` embed text
automatically through this function.
"""

import hashlib
import logging

import chromadb
import httpx

from ..config import settings
from ..models import KnowledgeResult

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Module-level client and collection (lazy-initialized)
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection (lazy singleton).

    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2, 384-dim)
    so both ``query(query_texts=...)`` and ``upsert(documents=...)`` embed text
    automatically — no explicit embedding call needed.
    """
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def set_collection_for_testing(collection: chromadb.Collection) -> None:
    """Override the collection for testing purposes."""
    global _collection
    _collection = collection


def reset_collection() -> None:
    """Reset the lazy singleton (useful after tests)."""
    global _client, _collection
    _collection = None
    _client = None


def _doc_id(text: str, source: str) -> str:
    """Generate a deterministic ID for a document."""
    return hashlib.sha256(f"{source}:{text[:200]}".encode()).hexdigest()[:16]


async def _brave_web_search(query: str) -> list[dict]:
    """Search Brave web API and return structured results."""
    if not settings.brave_api_key:
        return []

    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": settings.brave_api_key,
    }
    params = {"q": query, "count": 3}

    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(
            BRAVE_SEARCH_URL, headers=headers, params=params
        )
        response.raise_for_status()
        data = response.json()

    results: list[dict] = []
    for r in data.get("web", {}).get("results", []):
        result = KnowledgeResult(
            text=f"{r.get('title', '')}: {r.get('description', '')}",
            source=r.get("url", ""),
            title=r.get("title", ""),
        )
        results.append(result.model_dump())
    return results


def _store_in_chromadb(collection: chromadb.Collection, results: list[dict]) -> None:
    """Store web search results in ChromaDB for future retrieval."""
    if not results:
        return

    ids = [_doc_id(r["text"], r["source"]) for r in results]
    documents = [r["text"] for r in results]
    metadatas = [
        {"source": r["source"], "title": r.get("title", "")} for r in results
    ]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_local_knowledge(
    question: str, parse_result: dict | None = None
) -> str:
    """Query the local ChromaDB knowledge base (no web fallback).

    Strategy:
    1. Use knowledge_queries from parse result if available, else raw question
    2. Query ChromaDB for similar documents (semantic search, cosine distance)
    3. Filter by distance threshold
    4. Return formatted results for LLM consumption
    """
    parse_result = parse_result or {}
    collection = _get_collection()

    # Use parsed knowledge queries if available, otherwise raw question
    queries = parse_result.get("knowledge_queries") or [question]
    query_text = " ".join(queries) if len(queries) > 1 else queries[0]
    logger.debug(
        "Local knowledge query: %r (from %d sub-queries)",
        query_text[:100],
        len(queries),
    )

    results = collection.query(query_texts=[query_text], n_results=settings.kb_max_results)

    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Filter by distance threshold
    good_results: list[dict] = [
        KnowledgeResult(
            text=doc,
            source=meta.get("source", "local"),
            title=meta.get("title", ""),
        ).model_dump()
        for doc, dist, meta in zip(documents, distances, metadatas)
        if dist < settings.kb_max_distance
    ]

    logger.debug(
        "ChromaDB returned %d good results (threshold %.2f)",
        len(good_results),
        settings.kb_max_distance,
    )

    if not good_results:
        return "No relevant local knowledge found."

    parts: list[str] = []
    for i, r in enumerate(good_results, 1):
        source = r.get("source", "local")
        title = r.get("title", "")
        if source and source != "local":
            ref_label = f"[{title}]({source})" if title else source
            parts.append(f"[{i}] {r['text']}\n    Reference: {ref_label}")
        else:
            parts.append(f"[{i}] {r['text']}")

    return "\n\n".join(parts)


async def fetch_web_knowledge(
    question: str, parse_result: dict | None = None
) -> str:
    """Search the web for knowledge and store results in ChromaDB.

    Strategy:
    1. Build search query from parse result (news_query or knowledge_queries)
    2. Call Brave web search API
    3. Store results in ChromaDB for future local retrieval
    4. Return formatted results with source references
    """
    parse_result = parse_result or {}
    collection = _get_collection()

    # Build query from parse result
    queries = parse_result.get("knowledge_queries") or [question]
    query_text = " ".join(queries) if len(queries) > 1 else queries[0]
    fallback_query = parse_result.get("news_query") or query_text

    logger.debug("Web knowledge search: %r", fallback_query)
    web_results = await _brave_web_search(fallback_query)

    if not web_results:
        return "No relevant web results found."

    # Store in ChromaDB for future local queries
    logger.debug(
        "Web search returned %d results, storing in ChromaDB", len(web_results)
    )
    _store_in_chromadb(collection, web_results)

    # Format with reference URLs
    parts: list[str] = []
    for i, r in enumerate(web_results, 1):
        title = r.get("title", "")
        source = r.get("source", "")
        ref_label = f"[{title}]({source})" if (title and source) else source
        parts.append(f"[{i}] {r['text']}\n    Reference: {ref_label}")

    return "\n\n".join(parts)
