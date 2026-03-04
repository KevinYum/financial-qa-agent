"""Knowledge base tool — ChromaDB vector search with Brave web fallback.

Embedding: ChromaDB's default embedding function (all-MiniLM-L6-v2 via
onnxruntime, 384-dim). The model is bundled with chromadb — no extra
download or API key required. Both `query` and `upsert` embed text
automatically through this function.
"""

import hashlib

import chromadb
import httpx

from ..config import settings

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Module-level client and collection (lazy-initialized)
_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection (lazy singleton).

    Uses ChromaDB's default embedding function (all-MiniLM-L6-v2, 384-dim)
    so both `query(query_texts=...)` and `upsert(documents=...)` embed text
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
    """Fallback: search Brave web API and return structured results."""
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
        results.append({
            "text": f"{r.get('title', '')}: {r.get('description', '')}",
            "source": r.get("url", ""),
        })
    return results


def _store_in_chromadb(collection: chromadb.Collection, results: list[dict]) -> None:
    """Store web search results in ChromaDB for future retrieval."""
    if not results:
        return

    ids = [_doc_id(r["text"], r["source"]) for r in results]
    documents = [r["text"] for r in results]
    metadatas = [{"source": r["source"]} for r in results]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)


async def fetch_knowledge(question: str, parse_result: dict | None = None) -> str:
    """Query the knowledge base. Falls back to web search if results are sparse.

    Strategy:
    1. Use knowledge_queries from parse result if available, otherwise raw question
    2. Query ChromaDB for similar documents (semantic search)
    3. Filter by distance threshold (cosine similarity)
    4. If too few good results, search Brave and store for future use
    5. Return combined results formatted for LLM consumption
    """
    parse_result = parse_result or {}
    collection = _get_collection()

    # Use parsed knowledge queries if available, otherwise raw question
    queries = parse_result.get("knowledge_queries") or [question]
    query_text = " ".join(queries) if len(queries) > 1 else queries[0]

    results = collection.query(query_texts=[query_text], n_results=3)

    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Filter by distance threshold
    good_results: list[dict] = [
        {"text": doc, "source": meta.get("source", "local")}
        for doc, dist, meta in zip(documents, distances, metadatas)
        if dist < settings.kb_max_distance
    ]

    if len(good_results) < settings.kb_min_results:
        # Fallback to Brave web search
        fallback_query = parse_result.get("news_query") or query_text
        web_results = await _brave_web_search(fallback_query)
        if web_results:
            _store_in_chromadb(collection, web_results)
            good_results.extend(web_results)

    if not good_results:
        return "No relevant knowledge found for this question."

    parts: list[str] = []
    for i, r in enumerate(good_results, 1):
        source_label = f" (source: {r['source']})" if r["source"] != "local" else ""
        parts.append(f"[{i}] {r['text']}{source_label}")

    return "\n\n".join(parts)
