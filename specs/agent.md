# Agent Specification

**Version**: 0.9.0
**Last Updated**: 2026-03-04

## Overview
The financial QA agent uses a LangGraph `StateGraph` to orchestrate a multi-step pipeline: **parse** the user's question into structured entities, **fetch** relevant data from tools driven by the parse result, and **synthesize** a final answer via an LLM call. The knowledge path uses a "knowledge hub" approach: local ChromaDB first, then an LLM-gated web fallback if local results are insufficient.

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
    knowledge_query: str | None   # Single RAG-optimized search query for knowledge base
    needs_fundamentals: bool     # Whether to fetch FMP fundamental data (equity only)
    fundamental_endpoints: list[str]  # Which data sources: financial_statement, earnings_transcript

class AgentState(TypedDict):
    question: str                # Original user question
    parse_result: ParseResult    # Structured entities extracted by parse node
    question_type: str           # "analysis" (tickers present) or "knowledge" (no tickers)
    market_data: str             # Output from market data tool (empty if not called)
    fundamental_data: str        # Output from FMP fundamental data tool (empty if not called)
    news_data: str               # Output from news search tool (empty if not called)
    local_knowledge_data: str    # Output from local ChromaDB search (empty if not called)
    web_knowledge_data: str      # Output from web knowledge search (empty if not called)
    knowledge_sufficient: str    # "yes" | "no" — LLM evaluation of local knowledge sufficiency
    answer: str                  # Final synthesized answer
```

## Graph Topology

```
parse → route_by_parse_result (driven by question_type from parser)
         ├── analysis (+ tickers): fetch_market_data [+ fetch_fundamental_data] [+ fetch_news] → synthesize → END
         └── knowledge:            fetch_local_knowledge → evaluate_sufficiency
                                     ├── sufficient → synthesize → END
                                     └── insufficient → fetch_web_knowledge → synthesize → END
```

## Nodes

### 1. parse (LLM call)
- Receives the user question
- Makes a single LLM call to extract structured entities (tickers, time period, news flag, knowledge query, etc.)
- **Injects today's date** (`date.today().isoformat()`) into the prompt so the LLM can resolve relative/ambiguous dates correctly
- LLM JSON output is validated through `ParseResultModel` (Pydantic) which provides defaults for missing fields and catches type errors, then converted to dict via `.model_dump()` for LangGraph state compatibility
- **`time_period` constrained to valid yfinance values** — the prompt enumerates all 11 valid periods (`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`) and instructs the LLM to pick the smallest valid period that covers the requested range (e.g. "2 weeks" → `1mo`). `ParseResultModel` also has a `field_validator` that coerces invalid values to `None` as a safety net.
- **`time_start`/`time_end` for specific date queries** — the prompt instructs the LLM to use `time_start`/`time_end` (YYYY-MM-DD) instead of `time_period` when the user asks about a specific date or date range (e.g. "AAPL on 1/15" → `time_start="2026-01-15"`, `time_end="2026-01-16"`). `ParseResultModel` validates format via regex, coerces invalid formats to `None`, and a `model_validator` clears `time_period` when `time_start` is set (date range takes priority).
- No classification label — routing is driven entirely by which fields are populated
- Returns empty `parse_result: {}` on JSON parse failure (graceful fallback)
- Strips markdown code fences from LLM output before parsing

### 2. fetch_fundamental_data (FMP API)
- Triggered when `parse_result.needs_fundamentals` is `true` (analysis path only)
- **Equity-only**: Guards against non-equity asset types and ticker patterns (suffixes `-USD`, `=X`, `=F`; prefix `^`)
- Fetches from FMP free tier (250 req/day) via async httpx — **exactly 2 API calls max per ticker**
- Endpoints controlled by `parse_result.fundamental_endpoints` (only two valid values):
  - `financial_statement` → 1 FMP call to `/income-statement` (revenue, gross profit, operating income, net income, EPS, EBITDA)
  - `earnings_transcript` → 1 FMP call to `/earning-call-transcript`, summarized via LLM before returning
- **No default fallback** — if the parser doesn't set `fundamental_endpoints`, the tool returns early with a message. The parser is the explicit gatekeeper.
- Period inference: `time_period` short values (1d, 5d, 1mo, 3mo, 6mo) → quarterly; else → annual; limit=4 records
- Transcript handling: Transcripts are 5k-15k words. A dedicated LLM call (`_summarize_transcript`) condenses the transcript into ~300-500 words organized by: Financial Highlights, Guidance & Outlook, Strategic Updates, Risk Factors, Key Q&A Points — guided by the user's question for relevance
- Caps at 3 tickers per question (API budget)
- Graceful degradation: missing `FMP_API_KEY` → message, no tickers → message, non-equity → message
- Output: Formatted text with sections per ticker (Income Statement, Earnings Call Transcript Summary)

### 3. fetch_market_data (yfinance)
- Triggered when `parse_result.tickers` is non-empty (analysis or knowledge-with-tickers paths)
- Tickers come exclusively from parse result (LLM extraction) — no regex fallback
- The parse LLM handles all resolution: company names → tickers, sectors → ETFs, crypto/forex/index symbols, commodities → futures
- Supports equities (`AAPL`), crypto (`BTC-USD`), forex (`EURUSD=X`), indices (`^GSPC`), sector ETFs (`XLK`), commodities (`GC=F` gold, `CL=F` oil, `SI=F` silver, etc.)
- Time period from `parse_result.time_period` (validated against yfinance periods, defaults to `5d`)
- Supports explicit date ranges via `time_start`/`time_end`
- All OHLCV data points are passed to the LLM (no server-side truncation) — the synthesize prompt instructs the LLM to select only question-relevant facts for the response
- Fetches OHLCV and fundamentals (price, market cap, P/E, sector) via yfinance
- Runs yfinance in a thread pool (it's synchronous)
- Caps at 5 tickers per question

### 4. fetch_news (Brave Search API)
- Triggered when `parse_result.needs_news` is `true`
- Uses `parse_result.news_query` for a refined search query (falls back to raw question)
- Returns top 5 results with title, description, source URL, age
- Gracefully returns message if `BRAVE_API_KEY` is not set

### 5. fetch_local_knowledge (ChromaDB)
- Triggered on the knowledge path (no tickers in parse result)
- Also serves as fallback when no other tools are triggered
- Uses `knowledge_query` from parse result for ChromaDB search (single RAG-optimized query); falls back to raw question
- **Embedding**: ChromaDB's default embedding function (all-MiniLM-L6-v2 via onnxruntime, 384-dim, **256-token max**). The model silently truncates input beyond 256 tokens (~200 words). To work within this limit, documents are LLM-summarized before storage — the summary is stored as ChromaDB's `document` (embedded for ANN search).
- **Chunking (v0.0.32)**: Long documents are split into chunks of `kb_chunk_size` words before storage. Each chunk is independently summarized and stored as a separate ChromaDB entry. Metadata per chunk: `source`, `title`, `full_text` (chunk text), `chunk_id` ("0", "1", ...).
- **Retrieval**: Returns individual chunks (not full documents) via `metadatas["full_text"]`. Falls back to `document` text for backward compatibility with pre-v0.0.32 entries.
- **Chunk ID display**: Output includes `(chunk N)` labels for each retrieved chunk.
- **Reference deduplication**: When multiple chunks from the same source URL are retrieved, the `Reference:` line appears only on the first chunk — preventing duplicate URLs in the synthesize context.
- Filters by distance threshold (`kb_max_distance`, default 0.8)
- Returns formatted results with source metadata, or `"No relevant local knowledge found."`
- Output format: `[n] (chunk N) text\n    Reference: [Title](URL)` (Reference only on first chunk per source)

### 6. evaluate_sufficiency (LLM call)
- Triggered after `fetch_local_knowledge` completes
- Evaluates whether local knowledge results are sufficient to answer the user's question
- Uses `EVALUATE_SUFFICIENCY_PROMPT` with user question + local results
- LLM returns JSON `{"sufficient": true/false, "reason": "brief explanation"}`
- Returns `knowledge_sufficient: "yes"` or `"no"`
- On JSON parse failure → defaults to `"no"` (conservative: trigger web search)
- This is a short, cheap LLM call — simple binary decision

### 7. fetch_web_knowledge (Brave web search + ChromaDB storage)
- Triggered only when `evaluate_sufficiency` returns `"no"`
- Uses `knowledge_query` or `news_query` from parse result for Brave web search query
- Searches Brave web API (top 3 results)
- **Stores results in ChromaDB** for future local retrieval (auto-population)
- **Chunked storage (v0.0.32)**: Before storing, each result is split into chunks of `kb_chunk_size` words. Each chunk is independently summarized via `_summarize_for_embedding` (texts at or under `kb_summarize_limit` words skip the LLM call). Each chunk stored as a separate ChromaDB entry with metadata: `source`, `title`, `full_text` (chunk text), `chunk_id` ("0", "1", ...). On LLM failure, falls back to a truncated prefix.
- **Output**: Web results are displayed as-is (no chunk labels) since chunking is for future local retrieval. Format uses shared `_format_knowledge_results()` helper with reference deduplication.
- **URL/title retention**: Results include title and URL metadata, stored in ChromaDB. Output format: `[n] text\n    Reference: [Title](URL)` — enables the synthesize node to include clickable source references.
- Gracefully returns `"No relevant web results found."` if no API key or no results

### 8. synthesize (LLM call)
- Receives all non-empty tool outputs as context
- Uses `parse_result.question_type` to select the prompt (analysis or knowledge)
- Selects prompt based on derived question type:
  - **Analysis prompt** (market data present): Produces up to 3 sections with `##` headers:
    - `## Fact` — Question-relevant factual data only (not a raw dump of all available data). The prompt instructs the LLM to select data points that directly address the user's question and structure them as a lead-in to the analysis.
    - `## Analysis` — Analytical interpretation that directly answers the user's question, connecting facts to a clear takeaway (trends, comparisons, implications)
    - `## References` — News source links as `- [Title](URL)` bullets (omitted if no news)
  - **Knowledge prompt** (no market data): Produces 2 sections with `##` headers:
    - `## Answer` — Clear explanation of concepts
    - `## References` — Source links as `- [Title](URL)` bullets, or "Based on general financial knowledge"
- Frontend renders the full markdown output via `marked.js` (sanitized with DOMPurify) — `##` headers, links, lists, bold, tables, code all render natively
- Makes a single LLM call to produce the final answer
- Instructed not to fabricate data or URLs

## Routing Logic

Routing is driven by the parser LLM's `question_type` field. Two clean, mutually exclusive paths:

| Question Type | Condition | Tools | Answer Format |
|---|---|---|---|
| **Analysis** | `question_type=analysis` AND `tickers` non-empty | fetch_market_data (always) + fetch_fundamental_data (if `needs_fundamentals`) + fetch_news (if `needs_news`) | Fact + Analysis + References |
| **Knowledge** | `question_type=knowledge` OR no tickers | fetch_local_knowledge → evaluate_sufficiency → [fetch_web_knowledge] | Answer + References |

**Key behaviors**:
- **Parser decides**: The parse LLM determines `question_type`. Questions about specific tickers' data, performance, reports → analysis. General concepts, history, education → knowledge.
- **Knowledge ignores tickers**: If `question_type=knowledge`, any tickers in the parse result are ignored — no market data or fundamental data is fetched.
- **Analysis requires tickers**: If `question_type=analysis` but no tickers extracted, falls back to knowledge path (can't analyse without data).
- Empty parse result → knowledge path (fallback)
- Analysis path can fan out to market_data + fundamental_data + news in parallel (LangGraph fan-out)

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
| `FMP_API_KEY` | (optional) | Financial Modeling Prep API (fundamental data, 250 req/day free tier) |
| `BRAVE_API_KEY` | (optional) | Brave Search API access |
| `CHROMA_PERSIST_DIR` | `data/chroma` | ChromaDB storage directory |
| `CHROMA_COLLECTION_NAME` | `financial_knowledge` | ChromaDB collection name |
| `KB_MAX_RESULTS` | `3` | Max documents to fetch from ChromaDB per query |
| `KB_MAX_DISTANCE` | `0.8` | Max cosine distance for "good" results |
| `KB_CHUNK_SIZE` | `5000` | Max words per chunk for ChromaDB storage |
| `KB_SUMMARIZE_LIMIT` | `200` | Word threshold for LLM summarization before embedding |

## Data Models

All structured data flowing through the pipeline is formalized in `src/financial_qa_agent/models.py` as Pydantic models:

- **Market data**: `HistoryRecord`, `TickerData` (with `Field(alias="52w_high")` for Python-invalid keys)
- **Fundamental data**: `IncomeStatementRecord`, `FundamentalData` (container with ticker, name, income statements, transcript summary)
- **Knowledge base**: `KnowledgeResult`
- **News search**: `NewsResult`
- **Parse validation**: `ParseResultModel` — companion to `ParseResult` TypedDict; validates LLM output with defaults. Includes `field_validator` for `time_period` (coerce invalid to `None`), `field_validator` for `time_start`/`time_end` (enforce YYYY-MM-DD regex), `model_validator` to clear `time_period` when `time_start` is set, `knowledge_query: str | None` (single RAG-optimized query replacing the old `knowledge_queries` list), `needs_fundamentals: bool` + `fundamental_endpoints: list[str]` with validator filtering invalid endpoint names against `VALID_FUNDAMENTAL_ENDPOINTS` (`"financial_statement"`, `"earnings_transcript"`).
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
| `fetch_fundamental_data` | `trace(started)` → `tool_input(fundamental_data)` → `tool_output(fundamental_data)` → `trace(completed)` |
| `fetch_news` | `trace(started)` → `tool_input(news_search)` → `tool_output(news_search)` → `trace(completed)` |
| `fetch_local_knowledge` | `trace(started)` → `tool_input(local_knowledge)` → `tool_output(local_knowledge)` → `trace(completed)` |
| `evaluate_sufficiency` | `trace(started)` → `trace(completed, detail=sufficient/insufficient + reason)` |
| `route_sufficiency` | `trace(completed, detail=skip web / fetch web)` |
| `fetch_web_knowledge` | `trace(started)` → `tool_input(web_knowledge)` → `tool_output(web_knowledge)` → `trace(completed)` |
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
