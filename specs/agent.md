# Agent Specification

**Version**: 0.2.0
**Last Updated**: 2026-03-04

## Overview
The financial QA agent uses a LangGraph `StateGraph` to orchestrate a three-step pipeline: **parse** the user's question into structured entities, **fetch** relevant data from tools driven by the parse result, and **synthesize** a final answer via an LLM call.

## State Schema

```python
class ParseResult(TypedDict, total=False):
    tickers: list[str]           # yfinance symbols: ["AAPL", "BTC-USD", "^GSPC"]
    company_names: list[str]     # Human-readable: ["Apple", "Bitcoin"]
    time_period: str | None      # yfinance period: "1d", "5d", "1mo", "3mo", etc.
    time_start: str | None       # Explicit date range start (YYYY-MM-DD)
    time_end: str | None         # Explicit date range end (YYYY-MM-DD)
    asset_type: str | None       # "equity", "etf", "crypto", "forex", "index", "sector"
    sector: str | None           # Sector name when asset_type is "sector"
    needs_news: bool             # Whether to fetch news
    news_query: str | None       # Refined search query for Brave
    knowledge_queries: list[str] # Conceptual sub-questions for knowledge base

class AgentState(TypedDict):
    question: str          # Original user question
    parse_result: ParseResult  # Structured entities extracted by parse node
    market_data: str       # Output from market data tool (empty if not called)
    news_data: str         # Output from news search tool (empty if not called)
    knowledge_data: str    # Output from knowledge base tool (empty if not called)
    answer: str            # Final synthesized answer
```

## Graph Topology

```
parse → route_by_parse_result → fetch_market_data ──┐
                               → fetch_news ──────────┤→ synthesize → END
                               → fetch_knowledge ─────┘
```

## Nodes

### 1. parse (LLM call)
- Receives the user question
- Makes a single LLM call to extract structured entities (tickers, time period, news flag, knowledge queries, etc.)
- No classification label — routing is driven entirely by which fields are populated
- Returns empty `parse_result: {}` on JSON parse failure (graceful fallback)
- Strips markdown code fences from LLM output before parsing

### 2. fetch_market_data (yfinance)
- Triggered when `parse_result.tickers` is non-empty
- Ticker resolution order: parse result → regex fallback → sector ETF mapping
- Supports equities (`AAPL`), crypto (`BTC-USD`), forex (`EURUSD=X`), indices (`^GSPC`), sector ETFs (`XLK`)
- Time period from `parse_result.time_period` (validated against yfinance periods, defaults to `5d`)
- Supports explicit date ranges via `time_start`/`time_end`
- Fetches OHLCV (last 10 data points) and fundamentals (price, market cap, P/E, sector) via yfinance
- Runs yfinance in a thread pool (it's synchronous)
- Caps at 5 tickers per question

### 3. fetch_news (Brave Search API)
- Triggered when `parse_result.needs_news` is `true`
- Uses `parse_result.news_query` for a refined search query (falls back to raw question)
- Returns top 5 results with title, description, source URL, age
- Gracefully returns message if `BRAVE_API_KEY` is not set

### 4. fetch_knowledge (ChromaDB + Brave fallback)
- Triggered when `parse_result.knowledge_queries` is non-empty
- Also serves as fallback when no other tools are triggered
- Uses `knowledge_queries` from parse result for ChromaDB search (joins multiple queries); falls back to raw question
- **Embedding**: ChromaDB's default embedding function (all-MiniLM-L6-v2 via onnxruntime, 384-dim). Both `query(query_texts=...)` and `upsert(documents=...)` embed text automatically — no explicit embedding call or external API needed.
- Filters by distance threshold (`kb_max_distance`, default 0.5)
- If fewer than `kb_min_results` (default 2) good results:
  - Falls back to Brave web search (top 3), using `news_query` or knowledge query text
  - Stores results in ChromaDB for future queries (auto-population)
- Returns combined results

### 5. synthesize (LLM call)
- Receives all non-empty tool outputs as context
- Makes a single LLM call to produce the final answer
- Instructed not to fabricate data

## Routing Logic

Routing is **data-driven** — no classification label. The `route_by_parse_result` function checks which fields are populated in the parse result:

| Parse Result Field | Tool Triggered | Condition |
|---|---|---|
| `tickers` (non-empty) | fetch_market_data | Any financial instrument identified |
| `needs_news` (true) | fetch_news | Question involves recent events/sentiment |
| `knowledge_queries` (non-empty) | fetch_knowledge | Conceptual/educational sub-questions |
| Nothing populated | fetch_knowledge | Fallback for general questions |

Multiple tools can fire in parallel when multiple fields are populated (LangGraph fan-out).

**Shared fields**: When multiple tickers are parsed, `time_period` uses the longest applicable span. `needs_news` is `true` if any ticker in the question warrants news retrieval.

## LLM Configuration

Configured via environment variables / `.env`:

| Variable | Default | Purpose |
|---|---|---|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `LLM_API_KEY` | (required) | Authentication |
| `LLM_MODEL` | `gpt-4o-mini` | Model identifier |

**OpenRouter**: Set `LLM_BASE_URL=https://openrouter.ai/api/v1` and use OpenRouter model names (e.g., `openai/gpt-4o-mini`). Extra headers (`HTTP-Referer`, `X-Title`) are auto-added when the URL contains "openrouter".

## Tool Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BRAVE_API_KEY` | (optional) | Brave Search API access |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB storage directory |
| `CHROMA_COLLECTION_NAME` | `financial_knowledge` | ChromaDB collection name |
| `KB_MIN_RESULTS` | `2` | Min good results before web fallback |
| `KB_MAX_DISTANCE` | `0.5` | Max cosine distance for "good" results |

## Public Interface

```python
class FinancialQAAgent:
    async def ask(self, question: str) -> str
```

The interface is unchanged from the original stub. `main.py` requires no modifications.
