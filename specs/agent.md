# Agent Specification

**Version**: 0.5.0
**Last Updated**: 2026-03-04

## Overview
The financial QA agent uses a LangGraph `StateGraph` to orchestrate a three-step pipeline: **parse** the user's question into structured entities, **fetch** relevant data from tools driven by the parse result, and **synthesize** a final answer via an LLM call.

## State Schema

```python
class ParseResult(TypedDict, total=False):
    tickers: list[str]           # yfinance symbols: ["AAPL", "BTC-USD", "^GSPC"]
    company_names: list[str]     # Human-readable: ["Apple", "Bitcoin"]
    time_period: str | None      # yfinance period — MUST be one of: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
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
    question_type: str     # "analysis" (tickers present) or "knowledge" (no tickers)
    market_data: str       # Output from market data tool (empty if not called)
    news_data: str         # Output from news search tool (empty if not called)
    knowledge_data: str    # Output from knowledge base tool (empty if not called)
    answer: str            # Final synthesized answer
```

## Graph Topology

```
parse → route_by_parse_result
         ├── tickers found (analysis):  fetch_market_data [+ fetch_news] → synthesize → END
         └── no tickers (knowledge):    fetch_knowledge                  → synthesize → END
```

## Nodes

### 1. parse (LLM call)
- Receives the user question
- Makes a single LLM call to extract structured entities (tickers, time period, news flag, knowledge queries, etc.)
- LLM JSON output is validated through `ParseResultModel` (Pydantic) which provides defaults for missing fields and catches type errors, then converted to dict via `.model_dump()` for LangGraph state compatibility
- **`time_period` constrained to valid yfinance values** — the prompt enumerates all 11 valid periods (`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`) and instructs the LLM to pick the smallest valid period that covers the requested range (e.g. "2 weeks" → `1mo`). `ParseResultModel` also has a `field_validator` that coerces invalid values to `None` as a safety net.
- No classification label — routing is driven entirely by which fields are populated
- Returns empty `parse_result: {}` on JSON parse failure (graceful fallback)
- Strips markdown code fences from LLM output before parsing

### 2. fetch_market_data (yfinance)
- Triggered when `parse_result.tickers` is non-empty (analysis path)
- Tickers come exclusively from parse result (LLM extraction) — no regex fallback
- The parse LLM handles all resolution: company names → tickers, sectors → ETFs, crypto/forex/index symbols
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
- Triggered on the knowledge path (no tickers in parse result)
- Also serves as fallback when no other tools are triggered
- Uses `knowledge_queries` from parse result for ChromaDB search (joins multiple queries); falls back to raw question
- **Embedding**: ChromaDB's default embedding function (all-MiniLM-L6-v2 via onnxruntime, 384-dim). Both `query(query_texts=...)` and `upsert(documents=...)` embed text automatically — no explicit embedding call or external API needed.
- Filters by distance threshold (`kb_max_distance`, default 0.5)
- If fewer than `kb_min_results` (default 2) good results:
  - Falls back to Brave web search (top 3), using `news_query` or knowledge query text
  - Stores results in ChromaDB for future queries (auto-population)
- **URL/title retention**: Brave web search results include title and URL metadata, stored in ChromaDB alongside document text. Output format: `[n] text\n    Reference: [Title](URL)` — enables the synthesize node to include clickable source references in knowledge answers.
- Returns combined results with reference metadata

### 5. synthesize (LLM call)
- Receives all non-empty tool outputs as context
- **Derives question type from data present** — `market_data` non-empty → analysis, otherwise → knowledge. This is independent of the `question_type` field in state (routing function state mutations don't propagate in LangGraph).
- Selects prompt based on derived question type:
  - **Analysis prompt** (market data present): Produces up to 3 sections with `##` headers:
    - `## Fact` — Structured factual data (prices, changes, fundamentals, metrics)
    - `## Analysis` — Analytical interpretation (trends, comparisons, implications)
    - `## References` — News source links as `- [Title](URL)` bullets (omitted if no news)
  - **Knowledge prompt** (no market data): Produces 2 sections with `##` headers:
    - `## Answer` — Clear explanation of concepts
    - `## References` — Source links as `- [Title](URL)` bullets, or "Based on general financial knowledge"
- Frontend renders the full markdown output via `marked.js` (sanitized with DOMPurify) — `##` headers, links, lists, bold, tables, code all render natively
- Makes a single LLM call to produce the final answer
- Instructed not to fabricate data or URLs

## Routing Logic

Routing determines the **question type** based on whether tickers are present in the parse result. Two mutually exclusive paths:

| Question Type | Condition | Tools | Answer Format |
|---|---|---|---|
| **Analysis** | `tickers` non-empty | fetch_market_data (always) + fetch_news (if `needs_news`) | Fact + Analysis + References |
| **Knowledge** | No tickers | fetch_knowledge | Answer + References |

The routing function sets `question_type` on the state dict for trace logging, but this mutation does not propagate to downstream nodes in LangGraph. The synthesize node independently derives the question type by checking whether `market_data` is non-empty.

**Key behaviors**:
- `needs_news` without tickers → knowledge path (not news)
- `knowledge_queries` with tickers → analysis path (knowledge queries ignored)
- Empty parse result → knowledge path (fallback)
- Analysis path can fan out to market_data + news in parallel (LangGraph fan-out)

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

## Data Models

All structured data flowing through the pipeline is formalized in `src/financial_qa_agent/models.py` as Pydantic models:

- **Market data**: `HistoryRecord`, `TickerData` (with `Field(alias="52w_high")` for Python-invalid keys)
- **Knowledge base**: `KnowledgeResult`
- **News search**: `NewsResult`
- **Parse validation**: `ParseResultModel` — companion to `ParseResult` TypedDict; validates LLM output with defaults
- **Trace events**: `TraceEvent`, `ToolInputEvent`, `ToolOutputEvent`, `AnswerEvent`, `ErrorEvent`

**Key constraint**: LangGraph requires `AgentState` and `ParseResult` to be `TypedDict`. Pydantic models serve as validation companions — construct model, call `.model_dump()`, insert into state. Tool functions still accept `parse_result: dict | None` for compatibility.

## Trace Events

The agent emits trace events during execution via `contextvars.ContextVar`-based infrastructure. This enables real-time SSE streaming without modifying node signatures. Trace events are constructed using Pydantic models (`TraceEvent`, `ToolInputEvent`, `ToolOutputEvent`, `AnswerEvent`, `ErrorEvent`) then serialized via `.model_dump()` before being pushed to the queue.

### Infrastructure

- **`_trace_queue`** (`ContextVar[asyncio.Queue | None]`): Task-local queue, default `None` (no-op when unused)
- **`_emit_trace(event_type, **kwargs)`**: Async helper — puts event on queue if set
- **`_emit_trace_sync(event_type, **kwargs)`**: Sync variant using `put_nowait` (for `route_by_parse_result`)

### Event Flow

| Stage | Events Emitted |
|---|---|
| `parse` | `trace(started)` → `trace(completed, detail=ticker/news/knowledge summary)` |
| `route` | `trace(completed, detail=tool list)` |
| `fetch_market_data` | `trace(started)` → `tool_input(market_data)` → `tool_output(market_data)` → `trace(completed)` |
| `fetch_news` | `trace(started)` → `tool_input(news_search)` → `tool_output(news_search)` → `trace(completed)` |
| `fetch_knowledge` | `trace(started)` → `tool_input(knowledge_base)` → `tool_output(knowledge_base)` → `trace(completed)` |
| `synthesize` | `trace(started)` → `answer(answer=text)` → `trace(completed)` |

### Sentinel

A `None` value is always pushed to the queue in the `finally` block of `ask_with_trace`, signaling the SSE generator to end the stream.

## Public Interface

```python
class FinancialQAAgent:
    async def ask(self, question: str) -> str
    async def ask_with_trace(self, question: str, trace_queue: asyncio.Queue) -> str
```

- **`ask(question)`**: Batch response — no trace events emitted (contextvars no-op). Original interface, unchanged.
- **`ask_with_trace(question, trace_queue)`**: Same pipeline, but sets the trace queue context var so all nodes emit events. Always pushes `None` sentinel to queue in `finally`.
