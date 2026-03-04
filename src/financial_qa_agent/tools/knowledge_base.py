"""Knowledge base tool — ChromaDB local search and Brave web search.

Two public functions:
- ``fetch_local_knowledge`` — query ChromaDB only (semantic vector search)
- ``fetch_web_knowledge``  — query Brave web API, store results in ChromaDB

**Embedding**: ChromaDB's default function (all-MiniLM-L6-v2 via onnxruntime,
384-dim, **256-token max**). The model silently truncates input beyond 256
tokens (~200 words).

**Chunking** (v0.0.32): Long documents are split into chunks of
``kb_chunk_size`` words before storage. Each chunk is independently
LLM-summarized (if exceeding ``kb_summarize_limit`` words) and stored as a
separate ChromaDB entry. Retrieval returns individual chunks — not full
documents — with chunk IDs visible in trace output. Duplicate references
from the same source URL are merged so the synthesize LLM sees each URL once.

**Storage per chunk**: ChromaDB ``document`` = LLM summary (for embedding),
``metadatas["full_text"]`` = chunk text, ``metadatas["chunk_id"]`` = "0"|"1"|…
"""

import hashlib
import logging

import chromadb
import httpx
from langchain_core.messages import HumanMessage

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


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks of at most *chunk_size* words.

    Short texts (at or under *chunk_size* words) return a single-element list.
    Uses simple word-boundary splitting.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks: list[str] = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks


def _chunk_id(text: str, source: str, chunk_index: int) -> str:
    """Generate a deterministic ID for a document chunk.

    Format: ``{base_hash}_chunk_{n}`` where *base_hash* is derived from
    the original document's source + text prefix.
    """
    base = hashlib.sha256(f"{source}:{text[:200]}".encode()).hexdigest()[:16]
    return f"{base}_chunk_{chunk_index}"


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


SUMMARIZE_FOR_EMBEDDING_PROMPT = """\
Summarize the following text in under {limit} words. Preserve key facts, names, \
numbers, dates, and domain-specific terminology so that the summary works \
well as a search index entry for semantic retrieval.

--- Text ---
{text}
"""


async def _summarize_for_embedding(text: str) -> str:
    """Summarize text to fit within the embedding model's 256-token window.

    The all-MiniLM-L6-v2 model silently truncates beyond 256 tokens (~200
    words). For short texts that already fit, this is a no-op. For longer
    texts, an LLM call produces a concise summary optimized for semantic
    search.

    On LLM failure, falls back to a truncated prefix (better than nothing).
    """
    if len(text.split()) <= settings.kb_summarize_limit:
        return text

    # Import here to avoid circular import at module level
    from ..agent import _build_llm

    try:
        llm = _build_llm()
        prompt = SUMMARIZE_FOR_EMBEDDING_PROMPT.format(
            text=text[:15000], limit=settings.kb_summarize_limit
        )
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as exc:
        logger.warning("LLM summarization for embedding failed: %s", exc)
        return text[:1000]


async def _store_in_chromadb(
    collection: chromadb.Collection, results: list[dict]
) -> None:
    """Store web search results in ChromaDB, chunking long documents.

    Each result is split into chunks of ``kb_chunk_size`` words. Each chunk
    gets its own ChromaDB entry with an independently generated summary
    (for embedding) and the chunk text in ``metadatas["full_text"]``.
    """
    if not results:
        return

    all_ids: list[str] = []
    all_summaries: list[str] = []
    all_metadatas: list[dict] = []

    for r in results:
        text = r["text"]
        source = r["source"]
        title = r.get("title", "")

        chunks = _chunk_text(text, settings.kb_chunk_size)

        for i, chunk in enumerate(chunks):
            cid = _chunk_id(text, source, i)
            summary = await _summarize_for_embedding(chunk)

            all_ids.append(cid)
            all_summaries.append(summary)
            all_metadatas.append({
                "source": source,
                "title": title,
                "full_text": chunk,
                "chunk_id": str(i),
            })

    collection.upsert(ids=all_ids, documents=all_summaries, metadatas=all_metadatas)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_knowledge_results(results: list[dict]) -> str:
    """Format knowledge results with chunk IDs and deduplicated references.

    Each result entry shows its chunk ID (when present). The ``Reference:``
    line is printed only on the *first* occurrence of each source URL,
    preventing duplicate references flowing into the synthesize LLM.
    """
    seen_sources: set[str] = set()
    parts: list[str] = []

    for i, r in enumerate(results, 1):
        source = r.get("source", "local")
        title = r.get("title", "")
        chunk_id = r.get("chunk_id")

        # Build chunk label
        chunk_label = f" (chunk {chunk_id})" if chunk_id is not None else ""

        # Build entry text
        entry = f"[{i}]{chunk_label} {r['text']}"

        # Add reference only on the first chunk from each source URL
        if source and source != "local" and source not in seen_sources:
            ref_label = f"[{title}]({source})" if title else source
            entry += f"\n    Reference: {ref_label}"
            seen_sources.add(source)

        parts.append(entry)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def fetch_local_knowledge(
    question: str, parse_result: dict | None = None
) -> str:
    """Query the local ChromaDB knowledge base (no web fallback).

    Strategy:
    1. Use knowledge_query from parse result if available, else raw question
    2. Query ChromaDB for similar documents (semantic search, cosine distance)
    3. Filter by distance threshold
    4. Return formatted results for LLM consumption
    """
    parse_result = parse_result or {}
    collection = _get_collection()

    # Use parsed knowledge query if available, otherwise raw question
    query_text = parse_result.get("knowledge_query") or question
    logger.debug("Local knowledge query: %r", query_text[:100])

    results = collection.query(
        query_texts=[query_text], n_results=settings.kb_max_results
    )

    distances = results.get("distances", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Filter by distance threshold, then cap at kb_max_results.
    # Use full_text from metadata when available (post-v0.0.31 entries store
    # the LLM summary as document and original text/chunk text as full_text).
    # Fall back to document text for backward compatibility with older entries.
    good_results: list[dict] = []
    for doc, dist, meta in zip(documents, distances, metadatas):
        if dist < settings.kb_max_distance:
            good_results.append({
                "text": meta.get("full_text") or doc,
                "source": meta.get("source", "local"),
                "title": meta.get("title", ""),
                "chunk_id": meta.get("chunk_id"),  # None for pre-v0.0.32 entries
            })
        if len(good_results) >= settings.kb_max_results:
            break

    logger.debug(
        "ChromaDB returned %d good results (threshold %.2f)",
        len(good_results),
        settings.kb_max_distance,
    )

    if not good_results:
        return "No relevant local knowledge found."

    return _format_knowledge_results(good_results)


async def fetch_web_knowledge(
    question: str, parse_result: dict | None = None
) -> str:
    """Search the web for knowledge and store results in ChromaDB.

    Strategy:
    1. Build search query from parse result (knowledge_query or news_query)
    2. Call Brave web search API
    3. Store results in ChromaDB for future local retrieval
    4. Return formatted results with source references
    """
    parse_result = parse_result or {}
    collection = _get_collection()

    # Build query from parse result
    query_text = parse_result.get("knowledge_query") or question
    fallback_query = parse_result.get("news_query") or query_text

    logger.debug("Web knowledge search: %r", fallback_query)
    web_results = await _brave_web_search(fallback_query)

    if not web_results:
        return "No relevant web results found."

    # Store in ChromaDB for future local queries
    logger.debug(
        "Web search returned %d results, storing in ChromaDB", len(web_results)
    )
    await _store_in_chromadb(collection, web_results)

    # Format with reference URLs (web results are shown as-is; chunking
    # happens during storage for future local retrieval, so no chunk_id here).
    formatted_results = [
        {
            "text": r["text"],
            "source": r.get("source", ""),
            "title": r.get("title", ""),
            "chunk_id": None,
        }
        for r in web_results
    ]
    return _format_knowledge_results(formatted_results)
