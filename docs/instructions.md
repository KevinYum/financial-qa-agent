# Instruction Log

All user instructions that drive project direction are recorded here with timestamps and version numbers.

---

## v0.0.1 — 2026-03-04T05:24:08Z — Project Initialization

**Instruction**:
> Initialize this workspace using claude standard with rule and spec.
> - Ongoing updates should always update related rules and spec created for claude.
> - Always test your code after each prompt, this should be in standard rule.
> - All instructions should be saved in a standalone place to track how this project is created, with proper timestamp and version appended.
> - This project includes backend and a web page front. Use Python and uv package manager for backend, use simple web SDK as it is for demo purpose, can easily start on localhost.
> - The backend acts like a financial QA agent, and basically has only 1 product API — to receive user financial questions. Agent behavior to be defined later.

**Actions Taken**:
- Created `CLAUDE.md` with project rules and development standards
- Created `specs/architecture.md` — system architecture overview
- Created `specs/api.md` — API endpoint specification
- Created `docs/instructions.md` — this instruction log
- Initialized Python project with `uv` and FastAPI
- Created backend skeleton with `/api/ask` and `/health` endpoints
- Created frontend demo page (vanilla HTML/CSS/JS)
- Verified full stack runs on localhost

---

## v0.0.2 — 2026-03-04T05:24:08Z — README with Architecture Diagrams

**Instruction**:
> I need a README created and maintained along with the project.
> Specially, I will need this README include 3 architecture graphs:
> - One is for the entire project architecture, front/backend/middleware component and dataflow, interaction pattern
> - Second one is the design of agent loop to generate the reply for user input
> - Third one is the folder or major file content

**Actions Taken**:
- Created `README.md` with quick start, API reference, and dev commands
- Added 3 Mermaid diagrams:
  1. **System Architecture** — Browser, FastAPI, middleware (CORS, StaticFiles), routes, Pydantic models, agent core, full data flow
  2. **Agent Loop** — validation → empty check → processing pipeline (TBD: parse, retrieve, generate, format) → response builder
  3. **Project Structure** — all directories and files with responsibilities and relationships
- Updated `CLAUDE.md` rule #2 to require README diagram maintenance on changes
- Updated `CLAUDE.md` project structure tree to include `README.md`
- Updated `specs/architecture.md` to cross-reference README diagrams
- Removed stale `main.py` from project root (leftover from `uv init`)

---

## v0.0.3 — 2026-03-04T05:24:08Z — Project Structure Fix + Agent Loop Design

**Instruction**:
> 1. Project structure should be text-based like `folderA/... --- what is the content or file usage`, not in diagram.
> 2. Let's start design the agent loop. Do you think it's a good idea to orchestrate agent loop using LangGraph SDK, cuz it will be a multi-step fetch and then use all content fetched in different steps to generate the final answer through an LLM call.

**Actions Taken**:
- Replaced Mermaid project structure diagram in `README.md` with a text-based tree using `---` annotations
- Updated `CLAUDE.md` rule #2 to note diagram #3 is text-based (not Mermaid)
- Discussed LangGraph SDK as agent orchestrator — design decision pending

---

## v0.0.4 — 2026-03-04T05:24:08Z — LangGraph Agent Loop Implementation

**Instruction**:
> - Let's keep it pluggable but we can start with OpenAI, also the OpenRouter should be a valid option
> - I'm proposing to differentiate 3 types of data:
>   - Structured asset/equity data should be fetched from some data API, like OHLCV, fundamentals
>   - Unstructured asset/equity data like news should be fetched through web search (Brave API)
>   - Knowledge data should be fetched from local knowledge base (small vec db), if not enough it should be a web search and save it into the knowledge base for future use
> - Formalize the three fetch into tools and put under tools folder for clarity
> - Use yfinance first, Brave Search for news, ChromaDB for knowledge base

**Design Decisions**:
1. **LangGraph as orchestrator** — Chosen over plain async because the pipeline is a directed graph with conditional branching and parallel fan-out. LangGraph's `StateGraph` gives us a declarative topology (classify → route → fetch → synthesize) with built-in support for conditional edges and parallel execution on the `mixed` path. This also makes the pipeline easy to visualize and extend with new tools later.
2. **Classify-route-fetch-synthesize pattern** — A single LLM call classifies the question into one of 4 categories (`market_data`, `news`, `knowledge`, `mixed`). This avoids unnecessary tool calls (e.g., no yfinance for "What is compound interest?") and keeps latency low for single-tool questions. The `mixed` fallback triggers all 3 tools in parallel when the question spans multiple categories.
3. **4-way classification** — `market_data` (ticker-specific structured data), `news` (recent events/sentiment), `knowledge` (conceptual/educational), `mixed` (needs multiple sources). Invalid LLM responses default to `mixed` as a safe fallback.
4. **Pluggable LLM via `langchain-openai`** — `ChatOpenAI` supports both OpenAI and OpenRouter by swapping `base_url`. OpenRouter detection is automatic (checks for "openrouter" in the URL to add required headers). All config is externalized to `.env` via `pydantic-settings`.
5. **Tool design — 3 independent modules**:
   - `market_data.py`: Ticker extraction via regex (`$AAPL` and standalone uppercase words) with a stop-word filter for common English words. yfinance runs in a thread pool since it's synchronous. Caps at 3 tickers per question.
   - `news_search.py`: Brave Search API with `freshness=pw` (past week). Graceful degradation when API key is missing.
   - `knowledge_base.py`: ChromaDB (cosine similarity) as primary source. If fewer than `kb_min_results` good results (distance < `kb_max_distance`), falls back to Brave web search and **auto-stores results in ChromaDB** so the knowledge base grows over time.
6. **Shared state, not message passing** — `AgentState(TypedDict)` holds all intermediate data. Each fetch tool writes to its own field (`market_data`, `news_data`, `knowledge_data`). The synthesize node reads all non-empty fields. This avoids coupling between tools.
7. **Interface preservation** — `FinancialQAAgent.ask(question) -> str` kept identical to the original stub, so `main.py` required zero changes.

**Actions Taken**:
- Added dependencies: `langgraph`, `langchain-openai`, `langchain-core`, `chromadb`, `yfinance`, `httpx`, `pydantic-settings`
- Created `src/financial_qa_agent/config.py` — centralized Settings via pydantic-settings
- Created `.env.example` — template for LLM, Brave, ChromaDB env vars
- Created `src/financial_qa_agent/tools/` directory with 3 tool modules:
  - `market_data.py` — yfinance OHLCV + fundamentals, ticker extraction
  - `news_search.py` — Brave Search API for recent financial news
  - `knowledge_base.py` — ChromaDB vector search + Brave web fallback with auto-population
- Rewrote `src/financial_qa_agent/agent.py` — LangGraph StateGraph with 5 nodes (classify, 3 fetch, synthesize), conditional routing, parallel fan-out for "mixed"
- `main.py` unchanged — FinancialQAAgent interface preserved
- Updated `tests/test_api.py` — mocked agent for all API tests
- Created `tests/conftest.py` — shared test fixtures
- Created `tests/test_tools.py` — 11 unit tests for all 3 tool modules
- Created `tests/test_agent.py` — 6 integration tests for full LangGraph pipeline
- All 21 tests passing
- Updated `README.md` — all 3 diagrams (system arch, agent loop, project structure) + tech stack
- Updated `CLAUDE.md` — tech stack, project structure tree
- Created `specs/agent.md` — agent loop specification
- Updated `specs/architecture.md` — added agent, tools, external services sections
- Updated `specs/api.md` — noted LLM-powered pipeline
- Updated `.gitignore` — added `data/`

---

## v0.0.5 — 2026-03-04T05:24:08Z — Documentation Cleanup

**Instruction**:
> 1. Instruction log is missing major design decisions made during the agent loop implementation (v0.4.0).
> 2. Replace the README agent loop Mermaid diagram with a text-based diagram (same style as the project structure section — using box-drawing characters, not Mermaid).

**Actions Taken**:
- Added **Design Decisions** section to v0.0.4 entry documenting: LangGraph choice, classify-route-fetch-synthesize pattern, 4-way classification, pluggable LLM, tool design rationale, shared state approach, interface preservation
- Replaced agent loop Mermaid diagram in `README.md` with text-based box-drawing diagram
- Logged this instruction

---

## v0.0.6 — 2026-03-04T05:24:08Z — Versioning Fix + ChromaDB Embedding Clarification

**Instruction**:
> 1. All current instructions should be in version 0.0.x, will bump to 0.1.x when called out.
> 2. The current ChromaDB usage doesn't have embedding — how does insert and fetch work in this scenario?

**Actions Taken**:
- Renumbered all instruction versions from `v0.x.0` to `v0.0.x`
- Investigated ChromaDB embedding: it uses a **default embedding function** (all-MiniLM-L6-v2 via onnxruntime, 384-dim) that is bundled with the `chromadb` package. Both `query(query_texts=...)` and `upsert(documents=...)` embed text automatically — no explicit embedding call or external API needed. The code is functional as-is, but this was undocumented.
- Added module docstring to `knowledge_base.py` explaining the implicit embedding behavior
- Added docstring to `_get_collection()` explaining the default embedding function
- Updated `specs/agent.md` fetch_knowledge section to document the embedding model

---

## v0.0.7 — 2026-03-04T05:24:08Z — Parse-Driven Routing (Remove Classification)

**Instruction**:
> 1. The agent loop should have a different structure: before classify, you need to understand the user question using an LLM call and identify tickers, timespan, and knowledge questions.
> 2. After understanding the question, use 3 tools to retrieve data based on the parse result — the "type" classification is unnecessary, routing should depend on the parse result.
> 3. Parse result and tool results are both put into context, then passed to synthesize to produce the final answer.
> 4. News search should be triggered by a parse result flag (needs_news). Time span and news flag are shared across tickers — use the longest time span, true if any ticker needs news.

**Design Decisions**:
1. **Removed classification entirely** — The classify step was an artificial bottleneck. Questions often need multiple tools, and a single category label can't express that. Instead, the parse result itself determines which tools run: `tickers` → market data, `needs_news` → news, `knowledge_queries` → knowledge base.
2. **Parse-driven routing** — `route_by_parse_result` checks which fields are populated in the parse result. Multiple tools fire in parallel when multiple fields are populated. This is data-driven, not label-driven.
3. **ParseResult schema** — Replaces the old `classification` + `entities` structure with a flat TypedDict: `tickers`, `company_names`, `time_period`, `time_start/end`, `asset_type`, `sector`, `needs_news`, `news_query`, `knowledge_queries`. All fields are optional (total=False).
4. **Shared fields for multi-ticker queries** — When multiple tickers are mentioned, `time_period` uses the longest applicable span and `needs_news` is true if any ticker warrants news. This avoids per-ticker configuration complexity.
5. **Knowledge queries** — The parse LLM extracts conceptual sub-questions (e.g., "What are stock options?") which are used for ChromaDB search instead of the raw question. This improves retrieval relevance.
6. **Fallback behavior preserved** — Empty parse result falls back to fetch_knowledge (general questions still get answered). JSON parse failure returns empty parse_result, and all tools use the raw question as fallback.

**Actions Taken**:
- Rewrote `src/financial_qa_agent/agent.py`:
  - Removed `ExtractedEntities`, `classification` field, `CLASSIFY_PROMPT`, `classify_node`, `route_by_classification`
  - Added `ParseResult`, `PARSE_PROMPT`, `parse_node`, `route_by_parse_result`
  - Updated `AgentState`: replaced `classification` + `entities` with `parse_result`
  - Updated `build_graph`: renamed node from "classify" to "parse"
  - Updated `FinancialQAAgent.ask()`: initial state uses `parse_result: {}`
- Updated `src/financial_qa_agent/tools/knowledge_base.py`:
  - Uses `knowledge_queries` from parse result for ChromaDB search
  - Joins multiple knowledge queries for better retrieval
- Renamed `entities` param to `parse_result` across all 3 tool modules
- Rewrote `tests/test_agent.py`: 15 tests (pipeline, parse node, routing logic)
- Updated `tests/test_tools.py`: 21 tests (renamed kwargs, added knowledge_queries test)
- All 40 tests passing
- Rewrote `specs/agent.md` for parse → fetch → synthesize flow
- Updated `README.md`: agent loop diagram, system architecture, interaction pattern, test counts

---

## v0.0.8 — 2026-03-04T05:24:08Z — Trace Window with SSE Streaming

**Instruction**:
> Apart from the current main session window in front end, I also want you to add another trace window, this window have 4 tab:
> 1. First tab progressively showing the current question's agent loop process, like in which stage and this stage is doing what, it should be consistent with the debug log in backend side (btw, tweak the default log level into debug in backend)
> 2. 2,3,4 tab respectively shows the major input & output from different tools and should be reset after each new user question. For example on the market data tool, it should showing what are the query input including tickers, timespan, whether news, and also showing the api queried output. And trace output should be updated along with the project evolve.

**Design Decisions**:
1. **SSE (Server-Sent Events) for progressive updates** — The frontend needs real-time trace updates as each pipeline stage executes. SSE is simpler than WebSockets for this unidirectional stream. Since `EventSource` API is GET-only, the frontend uses `fetch()` with `ReadableStream` reader and parses SSE format manually from chunked responses.
2. **`contextvars.ContextVar` for trace queue** — LangGraph node functions have fixed `(state) -> dict` signatures. A context variable avoids modifying node signatures, is task-local for concurrent requests, and is a no-op when unused (existing `ask()` stays untouched).
3. **`asyncio.Queue` as event channel** — Trace events flow from agent nodes → queue → SSE generator. A `None` sentinel signals stream end. This decouples the agent execution from HTTP streaming.
4. **Separate SSE endpoint** — New `POST /api/ask/stream` alongside existing `POST /api/ask` (backward compatibility preserved). SSE endpoint registered before static file mount (catch-all).
5. **Two-panel frontend layout** — CSS Grid: chat panel (left) + trace panel (right) with 4 tabs (Agent Loop, Market Data, News Search, Knowledge Base). Responsive stacking below 900px.
6. **Agent Loop tab** — Progressive timeline with in-place status updates (⏳ started → ✓ completed). Each trace event updates the corresponding stage entry.
7. **Tool tabs** — Show tool input (query parameters) and output (fetched data) in monospace blocks. Reset on each new question.
8. **DEBUG logging in all tools** — Added `logging.getLogger(__name__)` and debug statements to market_data, news_search, and knowledge_base modules. Backend default log level set to DEBUG.

**Actions Taken**:
- Rewrote `src/financial_qa_agent/agent.py`:
  - Added contextvars-based trace infrastructure (`_trace_queue`, `_emit_trace`, `_emit_trace_sync`)
  - Instrumented all 5 nodes + routing function with trace events
  - Added `FinancialQAAgent.ask_with_trace(question, trace_queue)` method
  - Existing `ask()` unchanged — `_emit_trace` is a no-op without queue
- Rewrote `src/financial_qa_agent/main.py`:
  - Added `logging.basicConfig(level=logging.DEBUG)`
  - Added `POST /api/ask/stream` SSE endpoint with `StreamingResponse`
  - App version bumped to `0.2.0`
- Updated tool modules with DEBUG logging:
  - `src/financial_qa_agent/tools/market_data.py`
  - `src/financial_qa_agent/tools/news_search.py`
  - `src/financial_qa_agent/tools/knowledge_base.py`
- Rewrote `frontend/index.html` — two-panel layout with trace tabs
- Rewrote `frontend/style.css` — grid layout, tab styles, trace entry styles
- Rewrote `frontend/app.js` — SSE streaming, tab switching, trace rendering
- Updated `tests/test_api.py`: +3 SSE tests (content type, events, error case) → 7 total
- Updated `tests/test_agent.py`: +3 trace tests (event emission, empty question, no-op) → 18 total
- All 46 tests passing
- Updated `specs/api.md` — SSE endpoint with event type reference (v0.3.0)
- Updated `specs/agent.md` — Trace Events section, updated public interface (v0.3.0)
- Updated `specs/architecture.md` — SSE streaming, two-panel frontend, trace queue (v0.3.0)
- Updated `CLAUDE.md` — API design, frontend file descriptions
- Updated `README.md` — system architecture diagram (SSE flow), interaction pattern, project structure, API reference, test counts

---

## v0.0.9 — 2026-03-04T05:24:08Z — Two-Type Question Routing + Remove Regex Ticker Fallback

**Instruction**:
> 1. No longer need non-LLM fallback to parse tickers, we currently should only rely on parse node.
> 2. After the parse phase, we differentiate two types of question:
>    - Analysis for assets or markets
>    - General knowledge Q/A
> The indicator of type is whether there is a ticker to query live data later in tool calling. The type results in different answer format:
>    - Analysis: two paragraphs — fact data with structured format, and analysis result
>    - Knowledge: two paragraphs — the answer, and references including web search URLs
> URL retention requires changes to the knowledge base tool for Brave web search metadata.
>
> Also: fix the timestamp in instructions.md — use actual system time when appending, not fake incremental hours.

**Design Decisions**:
1. **Removed regex ticker fallback** — `_extract_tickers()` regex function, `_STOP_WORDS` frozenset, and `SECTOR_ETFS` dict deleted from `market_data.py`. The parse LLM is now the sole source of ticker resolution (company names → tickers, sectors → ETFs, crypto/forex/index symbols). This simplifies the code and eliminates the risk of regex false positives.
2. **Two-type question routing** — Questions are classified into exactly two types based on a single criterion: presence of tickers in the parse result. **Analysis** (tickers found) fetches market data + optional news, producing a data summary + analysis answer. **Knowledge** (no tickers) queries the knowledge base, producing an answer + references with URLs. This replaces the previous data-driven routing where any combination of tools could fire.
3. **Behavioral changes from old routing**:
   - `needs_news` without tickers → knowledge path (not news). The knowledge base falls back to Brave web search anyway, so news-style questions without specific tickers get answered via web search.
   - `knowledge_queries` with tickers → analysis path (knowledge queries ignored). When tickers are present, the focus is on market data analysis, not conceptual knowledge.
4. **Two type-specific synthesize prompts** — `SYNTHESIZE_ANALYSIS_PROMPT` instructs the LLM to produce (1) Data Summary with precise numbers, (2) Analysis with trends/context/news. `SYNTHESIZE_KNOWLEDGE_PROMPT` instructs (1) Answer with clear explanation, (2) References with `[Title](URL)` format.
5. **URL/title metadata retention** — The knowledge base tool now stores title and URL metadata from Brave web search results in ChromaDB. Output format changed from `[n] text (source: url)` to `[n] text\n    Reference: [Title](url)` for easy extraction by the synthesize LLM.
6. **State mutation in routing** — `question_type` is set directly on the state dict in `route_by_parse_result()`. **Note**: This mutation does not propagate to downstream nodes in LangGraph — fixed in v0.0.12 by having synthesize derive question type from data presence instead.

**Actions Taken**:
- Rewrote `src/financial_qa_agent/tools/market_data.py`:
  - Deleted `import re`, `_STOP_WORDS`, `SECTOR_ETFS`, `_extract_tickers()`
  - Simplified `fetch_market_data()` to use only parse_result tickers
- Updated `src/financial_qa_agent/tools/knowledge_base.py`:
  - `_brave_web_search()` now returns `title` field alongside `text` and `source`
  - `_store_in_chromadb()` stores title in metadata
  - `fetch_knowledge()` output format changed to `Reference: [Title](url)`
- Rewrote `src/financial_qa_agent/agent.py`:
  - Added `question_type: str` to `AgentState`
  - Replaced single `SYNTHESIZE_PROMPT` with `SYNTHESIZE_ANALYSIS_PROMPT` and `SYNTHESIZE_KNOWLEDGE_PROMPT`
  - Updated `synthesize_node()` to select prompt based on `question_type`
  - Rewrote `route_by_parse_result()` for two-type routing
  - Updated initial state in `ask()` and `ask_with_trace()`
- Updated `tests/test_tools.py`:
  - Deleted 7 regex-related tests (`TestExtractTickers`, `test_market_data_fetch_with_mock`, `test_market_data_entities_empty_falls_back_to_regex`)
  - Replaced `test_market_data_sector_fallback` → `test_market_data_sector_via_parse`
  - Added `test_market_data_empty_parse_result_no_tickers`, `test_knowledge_base_includes_reference_urls`
- Updated `tests/test_agent.py`:
  - Rewrote 6 routing tests with `question_type` assertions and new two-type behavior
  - Rewrote 2 pipeline tests for new routing (`test_agent_pipeline_news_without_tickers_goes_to_knowledge`, `test_agent_pipeline_tickers_with_news`)
  - Added 2 synthesize prompt selection tests
- All 44 tests passing (7 API + 21 agent + 16 tools)
- Updated `README.md` — agent loop diagram, interaction pattern, project structure (test counts, market_data description)
- Updated `specs/agent.md` — state schema, graph topology, routing logic, fetch_market_data, fetch_knowledge, synthesize sections (v0.4.0)
- Updated `specs/architecture.md` — overview, pipeline, data flow, tool descriptions (v0.4.0)

---

## v0.0.10 — 2026-03-04T05:39:18Z — Formalize Data Structures into Pydantic Models

**Instruction**:
> I want you to formalize the data structure in agent loop and also in tool calling and usage, they should be in structured and in model.py not in plain dict which makes it hard to maintain and understand.

**Design Decisions**:
1. **Centralized models.py** — All Pydantic models in a single file (`src/financial_qa_agent/models.py`) with no dependencies on other project modules. This is the single source of truth for all structured data shapes flowing through the pipeline.
2. **LangGraph TypedDict preserved** — `AgentState` and `ParseResult` remain as `TypedDict` in `agent.py` because LangGraph requires this. Pydantic models serve as validation companions: construct model → `.model_dump()` → insert into state.
3. **Alias handling for `52w_high`/`52w_low`** — These aren't valid Python identifiers. Solved with `Field(alias="52w_high")` + `ConfigDict(populate_by_name=True)`. Tools use `.model_dump(by_alias=True)` to preserve backward-compatible dict keys consumed by `_format_ticker_data()`.
4. **ParseResultModel for LLM validation** — The parse node now validates LLM JSON through `ParseResultModel(**parsed).model_dump()`, which provides sensible defaults for missing fields (e.g., `tickers=[]`, `needs_news=False`) and catches type errors from malformed LLM output. Extra fields from LLM are silently dropped (no `extra='forbid'`).
5. **Typed trace events** — `TraceEvent`, `ToolInputEvent`, `ToolOutputEvent`, `AnswerEvent`, `ErrorEvent` use `Literal` types for `event_type` fields. All trace emit call sites now use model constructors + `.model_dump()` spread for validation while keeping the generic `_emit_trace(**kwargs)` signature unchanged.
6. **Tool function signatures unchanged** — All tool functions still accept `parse_result: dict | None` for compatibility. Models are used internally for construction and validation, not as interface types.

**Actions Taken**:
- Created `src/financial_qa_agent/models.py` with 10 Pydantic models:
  - Market data: `HistoryRecord`, `TickerData`
  - Knowledge base: `KnowledgeResult`
  - News search: `NewsResult`
  - Parse validation: `ParseResultModel`
  - Trace events: `TraceEvent`, `ToolInputEvent`, `ToolOutputEvent`, `AnswerEvent`, `ErrorEvent`
- Updated `src/financial_qa_agent/tools/market_data.py` — imports and uses `HistoryRecord`, `TickerData`
- Updated `src/financial_qa_agent/tools/news_search.py` — imports and uses `NewsResult`
- Updated `src/financial_qa_agent/tools/knowledge_base.py` — imports and uses `KnowledgeResult`
- Updated `src/financial_qa_agent/agent.py` — imports `ParseResultModel` and all trace event models; parse node validates through `ParseResultModel`; all trace emit sites use model constructors
- Created `tests/test_models.py` — 20 tests covering all models (alias roundtrip, defaults, validation, trace event shapes)
- All 64 tests passing (7 API + 21 agent + 20 models + 16 tools)
- Updated `CLAUDE.md` — project structure (added `models.py`, `test_models.py`)
- Updated `README.md` — project structure (added `models.py`, `test_models.py`, test counts)
- Updated `specs/architecture.md` — added Data Models section, updated tool descriptions (v0.5.0)
- Updated `specs/agent.md` — added Data Models section, updated parse node and trace events descriptions (v0.5.0)

---

## v0.0.11 — 2026-03-04T05:50:36Z — Sectioned Answer Format with Markdown Rendering

**Instruction**:
> For the analysis and knowledge type, each different type's return has several paragraphs:
> 1. Each paragraph should have a user-visible delimiter, with bigger paragraph title like "Fact" and "Analysis"
> 2. It's correct that we may have news in the analysis part, which should be in another section called "References"
> 3. The front end when having URL is not correctly populated — URLs should be clickable
> 4. Markdown should be correctly populated in frontend, not forced into some custom HTML format

**Design Decisions**:
1. **Section-delimited answer format** — Replaced the old "two paragraphs separated by a blank line" instruction with explicit `##` markdown headers. The LLM now produces structured sections: `## Fact`, `## Analysis`, `## References` (analysis type) or `## Answer`, `## References` (knowledge type). The `## References` section is omitted entirely for analysis when no news data is available.
2. **Proper markdown rendering via marked.js** — Instead of a hand-rolled HTML converter that would need to handle every markdown feature, the frontend uses `marked.js` (CDN) to render LLM markdown output correctly. This handles `##` headers, `**bold**`, `[links](url)`, `- bullets`, paragraphs, tables, code blocks, and any other markdown the LLM produces — all natively.
3. **DOMPurify for sanitization** — All `marked.parse()` output is passed through `DOMPurify.sanitize()` before inserting into the DOM. This is defense-in-depth against any XSS that could come from LLM output. Links get `target="_blank"` and `rel="noopener noreferrer"` applied after sanitization.
4. **CSS scoped to `.msg.agent`** — Standard HTML elements (`h2`, `p`, `ul`, `li`, `a`, `strong`, `code`, `table`) are styled under `.msg.agent` to look correct within the chat bubble. `h2` gets the blue uppercase title treatment with border. Links are blue and underlined. User and error messages retain `white-space: pre-wrap`; agent messages use normal whitespace with HTML rendering.

**Actions Taken**:
- Rewrote `SYNTHESIZE_ANALYSIS_PROMPT` in `agent.py` — 3 sections: `## Fact`, `## Analysis`, `## References` (conditional)
- Rewrote `SYNTHESIZE_KNOWLEDGE_PROMPT` in `agent.py` — 2 sections: `## Answer`, `## References`
- Updated `frontend/index.html` — added `marked.min.js` and `purify.min.js` via CDN
- Updated `frontend/app.js`:
  - Configured `marked` with `breaks: true` and `gfm: true`
  - `addMessage()` now uses `marked.parse()` + `DOMPurify.sanitize()` for agent messages
  - Removed custom `renderAgentAnswer()` and `renderMarkdownBody()` — no longer needed
  - Links get `target="_blank"` applied after rendering
- Updated `frontend/style.css`:
  - Replaced custom `.answer-section*` and `.answer-bullet` classes with standard element styles under `.msg.agent` (`h2`, `p`, `ul`, `li`, `a`, `strong`, `code`, `table`)
  - Moved `white-space: pre-wrap` from `.msg` to `.msg.user, .msg.error` only
- Updated `tests/test_agent.py` — synthesize prompt assertions now check for `## Fact`, `## Analysis`, `## Answer`, `## References`
- All 64 tests passing
- Updated `specs/agent.md` — synthesize node and routing table reflect sectioned answer format
- Updated `README.md` — interaction pattern and agent loop diagram reflect sectioned answers with markdown rendering

---

## v0.0.12 — 2026-03-04T05:59:05Z — Fix Analysis Prompt Selection (question_type State Bug)

**Instruction**:
> The current analysis result is not following Fact, Analysis, optional Reference paragraph, but also uses the knowledge QA's Answer and Reference paragraph format. Find the reason and fix.

**Root Cause**:
The `route_by_parse_result()` function set `state["question_type"] = "analysis"` via direct dict mutation. However, LangGraph routing functions are used only to determine which nodes to route to — they don't return state updates like nodes do. Direct state mutations in a routing function are not propagated to subsequent nodes. When `synthesize_node()` ran `state.get("question_type", "knowledge")`, the mutation was lost and it always defaulted to `"knowledge"`, causing all questions to use the knowledge prompt format (`## Answer` + `## References`) instead of the analysis format (`## Fact` + `## Analysis` + `## References`).

**Design Decision**:
1. **Derive question type from data, not state field** — The synthesize node now determines question type by checking whether `market_data` is non-empty: `"analysis" if state.get("market_data") else "knowledge"`. This is reliable because `market_data` is set by `fetch_market_data_node()` which properly returns a state update dict. This eliminates the dependency on a routing-function state mutation that LangGraph doesn't propagate.
2. **Routing function mutation kept for tracing** — The `state["question_type"] = "analysis"` line in `route_by_parse_result()` is retained for trace event detail logging, but documented as non-authoritative. The synthesize node does not read it.

**Actions Taken**:
- Updated `src/financial_qa_agent/agent.py`:
  - `synthesize_node()`: replaced `state.get("question_type", "knowledge")` with `"analysis" if state.get("market_data") else "knowledge"`
  - Updated docstring explaining the derivation logic and why it's needed
  - `route_by_parse_result()`: updated docstring noting that state mutation is for tracing only
- Updated `tests/test_agent.py`:
  - `test_synthesize_uses_analysis_prompt`: `question_type` set to `""` (not `"analysis"`) to prove synthesize derives type from `market_data`
  - `test_synthesize_uses_knowledge_prompt`: `question_type` set to `""` (not `"knowledge"`) to prove synthesize derives type from empty `market_data`
- All 64 tests passing
- Updated `specs/agent.md` — synthesize node and routing logic sections reflect data-driven prompt selection

---

## v0.0.13 — 2026-03-04T06:31:32Z — Constrain time_period to Valid yfinance Values

**Instruction**:
> When parsing the time period, you should pass in all feasible period, or I will hit the case when I ask the past half month, the parse will produce "2w" as period but 2w is not feasible, it should return 1mo which contains 2w data point.

**Design Decisions**:
1. **Explicit enumeration in parse prompt** — The prompt now lists all 11 valid yfinance period values (`1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`) and explicitly states "No other values are valid — do NOT invent periods like 2w, 2d, 4mo, etc." with comprehensive mapping examples covering ambiguous cases (e.g., "past 2 weeks" → `1mo`, "9 months" → `1y`, "18 months" → `2y`).
2. **Round-up strategy** — The LLM is instructed to "always pick the smallest valid period that fully covers the requested range". For example, "past half month" (~15 days) → `1mo` (not `5d` which is too short, not `3mo` which is excessive). This ensures users always get data that covers their requested range.
3. **Pydantic field_validator as safety net** — Added a `field_validator("time_period")` on `ParseResultModel` that coerces any value not in `VALID_TIME_PERIODS` to `None`. This provides defense-in-depth: if the LLM still produces `"2w"`, the validator catches it and the downstream market_data tool falls back to its default period (`"5d"`). The `VALID_TIME_PERIODS` frozenset is defined once in `models.py` and mirrors the one in `market_data.py`.

**Actions Taken**:
- Updated `PARSE_PROMPT` in `src/financial_qa_agent/agent.py`:
  - Replaced loose time_period instructions with explicit enumeration of all valid values
  - Added "do NOT invent periods" constraint
  - Comprehensive mapping examples with round-up-to-cover semantics
- Updated `src/financial_qa_agent/models.py`:
  - Added `VALID_TIME_PERIODS` frozenset
  - Added `@field_validator("time_period")` on `ParseResultModel` to coerce invalid values to `None`
  - Added `field_validator` to Pydantic imports
- Added 3 tests in `tests/test_models.py`:
  - `test_valid_time_periods_accepted` — all 11 valid periods accepted
  - `test_invalid_time_period_coerced_to_none` — invalid values (`2w`, `4mo`, `2d`, etc.) coerced to `None`
  - `test_none_time_period_stays_none` — `None` preserved
- All 67 tests passing (7 API + 21 agent + 23 models + 16 tools)
- Updated `specs/agent.md` — parse node describes time_period constraint; state schema updated
- Updated `specs/architecture.md` — ParseResultModel description includes field_validator

---

## v0.0.14 — 2026-03-04T06:57:40Z — Knowledge Hub: LLM-Gated Local/Web Split

**Instruction**:
> In the knowledge route, you should differentiate what are the knowledge retrieved from local vec db and what are the online search result, and in the tab they should be in different tab, and also in the agent loop diagram, you should first fetch local and then ask llm to see whether local is enough to respond with user's question, if not kick off a web search, that's knowledge hub usage is that right?

**Design Decisions**:
1. **Knowledge Hub pattern (local-first, LLM-gated web fallback)** — The knowledge path now queries ChromaDB first (`fetch_local_knowledge`), then an LLM evaluates whether those results are sufficient to answer the question (`evaluate_sufficiency`). Only if the LLM says "no" does the agent trigger a web search (`fetch_web_knowledge`). This reduces unnecessary API calls and prioritizes local knowledge.
2. **Three new nodes (7-node graph)** — The graph grew from 5 to 7 nodes: `fetch_local_knowledge_node`, `evaluate_sufficiency_node`, `fetch_web_knowledge_node`. The evaluation node is a cheap LLM call that returns JSON `{"sufficient": true/false, "reason": "..."}`. On parse failure, it conservatively defaults to "no" (trigger web search).
3. **Two new state fields** — `AgentState` now has `local_knowledge_data`, `web_knowledge_data`, and `knowledge_sufficient` instead of the single `knowledge_data`. The synthesize node assembles context from both sources when available (`### Local Knowledge` + `### Web Search Results`).
4. **Second conditional routing edge** — `route_after_sufficiency(state)` routes to `"synthesize"` (sufficient) or `"fetch_web_knowledge"` (insufficient). This is the second conditional edge in the graph (first is `route_by_parse_result`).
5. **Separate trace tabs** — The single "Knowledge Base" tab is replaced by "Local Knowledge" and "Web Knowledge" tabs. The frontend now has 5 trace tabs. Tool trace events use `tool="local_knowledge"` and `tool="web_knowledge"` to map to the correct tabs.
6. **Split tool functions** — `knowledge_base.py` now exports two public functions: `fetch_local_knowledge` (ChromaDB only) and `fetch_web_knowledge` (Brave + ChromaDB storage). The old `fetch_knowledge` is removed. Internal helpers (`_brave_web_search`, `_store_in_chromadb`) are unchanged.

**Actions Taken**:
- Rewrote `src/financial_qa_agent/tools/knowledge_base.py`:
  - Replaced `fetch_knowledge` with `fetch_local_knowledge` and `fetch_web_knowledge`
  - `fetch_local_knowledge`: ChromaDB query + distance filter, no web fallback
  - `fetch_web_knowledge`: Brave web search + ChromaDB storage for future retrieval
- Updated `src/financial_qa_agent/tools/__init__.py` — new exports
- Rewrote `src/financial_qa_agent/agent.py`:
  - Updated `AgentState`: `local_knowledge_data`, `web_knowledge_data`, `knowledge_sufficient` replace `knowledge_data`
  - Added `EVALUATE_SUFFICIENCY_PROMPT`, `evaluate_sufficiency_node`, `fetch_local_knowledge_node`, `fetch_web_knowledge_node`
  - Added `route_after_sufficiency` routing function
  - Updated `build_graph()`: 7 nodes, two conditional edges
  - Updated `synthesize_node`: assembles context from both knowledge sources
  - Updated initial states in `ask()` and `ask_with_trace()`
- Updated `frontend/index.html` — 5 tabs: Agent Loop, Market Data, News Search, Local Knowledge, Web Knowledge
- Updated `frontend/app.js` — `TOOL_TAB_MAP` maps `local_knowledge` and `web_knowledge`; `resetTrace()` clears both new panes
- Updated `frontend/style.css` — reduced tab font/padding for 5 tabs
- Rewrote `tests/test_tools.py`:
  - Replaced 5 `fetch_knowledge` tests with 3 `fetch_local_knowledge` + 3 `fetch_web_knowledge` tests
- Rewrote `tests/test_agent.py`:
  - Split knowledge pipeline test into sufficient/insufficient variants
  - Added 3 `evaluate_sufficiency_node` tests (sufficient, insufficient, JSON failure)
  - Added 3 `route_after_sufficiency` tests (yes, no, missing defaults to no)
  - Added `test_synthesize_includes_both_knowledge_sources`
  - Updated all knowledge-path tests to use new state fields and mock both functions
- All 76 tests passing (7 API + 29 agent + 23 models + 17 tools)
- Updated `specs/agent.md` — 7-node graph topology, new state schema, knowledge hub nodes, sufficiency routing (v0.6.0)
- Updated `specs/architecture.md` — 7-node agent, knowledge hub pipeline, 5 trace tabs (v0.6.0)
- Updated `README.md` — system architecture diagram (7 nodes), agent loop diagram (knowledge hub), interaction pattern, project structure (test counts)
- Updated `CLAUDE.md` — knowledge_base.py description

---

## v0.0.15 — 2026-03-04 — Config Cleanup + Date Range Parsing Fix

**Instructions**:
> 1. "in the config, the kb_min_result is not used, actually I think the config should be a max result to fetch from vec db?"
> 2. "it seems when I query about past data like 1/15, the parse failed by returning time_period 1d, while time_start, time_end doesn't take effect, is that because you are just asking a time_period in parse node, maybe add and formating (make sure the format correct) the start / end time in parse would help? of course you need to make sure yfinance api support such past data query"

**Design Decisions**:
- **Config cleanup**: Replaced dead `kb_min_results: int = 2` (leftover from old web-fallback threshold logic removed in v0.0.14) with `kb_max_results: int = 3` to control how many documents ChromaDB returns per query. Wired into `knowledge_base.py` via `n_results=settings.kb_max_results`.
- **Date parsing fix**: Three-pronged approach:
  1. **Enhanced PARSE_PROMPT** — Expanded `time_start`/`time_end` instructions with explicit rules: use date range instead of `time_period` for specific dates, enforce YYYY-MM-DD format, single-date queries get `time_end` = next day, resolve ambiguous dates (no year) to most recent past occurrence. Added `Today's date: {today}` to prompt so LLM can resolve relative dates.
  2. **ParseResultModel validators** — `field_validator` on `time_start`/`time_end` enforces YYYY-MM-DD regex (coerces invalid formats to `None`). `model_validator` clears `time_period` when `time_start` is set (date range takes priority over period).
  3. **yfinance already supports date ranges** — `_fetch_ticker_data` in `market_data.py` already had `t.history(start=start, end=end)` logic, so no backend changes needed.

**Actions Taken**:
- Updated `src/financial_qa_agent/config.py` — `kb_min_results` → `kb_max_results` (default 3)
- Updated `src/financial_qa_agent/tools/knowledge_base.py` — `n_results=settings.kb_max_results`
- Updated `src/financial_qa_agent/agent.py`:
  - Added `from datetime import date` import
  - Enhanced `PARSE_PROMPT` with detailed `time_start`/`time_end` instructions and examples
  - Added `Today's date: {today}` to prompt template
  - Updated `parse_node` to format prompt with `today=date.today().isoformat()`
- Updated `src/financial_qa_agent/models.py`:
  - Added `re` and `model_validator` imports
  - Added `validate_date_format` field validator for `time_start`/`time_end` (YYYY-MM-DD regex)
  - Added `clear_period_when_dates_set` model validator (clears `time_period` when `time_start` is set)
- Added tests in `tests/test_models.py`:
  - `test_valid_date_formats_accepted` — YYYY-MM-DD accepted
  - `test_invalid_date_formats_coerced_to_none` — 7 invalid formats rejected
  - `test_none_dates_stay_none` — None preserved
  - `test_time_period_cleared_when_dates_set` — model validator works
  - `test_time_period_preserved_without_dates` — no false positive clearing
- Added tests in `tests/test_agent.py`:
  - `test_parse_node_date_range_query` — parse node extracts time_start/time_end
  - `test_parse_node_date_range_overrides_period` — model validator clears period
  - `test_agent_pipeline_date_range_query` — full pipeline with date range passes correct params to market_data tool
- Updated `specs/agent.md` — parse node docs (date handling, today's date injection, model validators)
- All 84 tests passing (7 API + 32 agent + 28 models + 17 tools)

---

## v0.0.16 — 2026-03-04 — Analysis Prompt: Question-Focused Fact Section

**Instruction**:
> in the fact section of the analysis route, it should not return exact the market data output which always the same, you should tweak the prompt so that the fact should correctly respond to user's concern and as a lead to the following analysis section

**Design Decision**:
The old `## Fact` prompt said "present key factual data… include current prices, price changes, fundamentals" — a generic instruction that caused the LLM to dump all available market data regardless of what the user asked. The new prompt instructs the LLM to:
1. **Select only question-relevant data points** — if the user asks about price, lead with price; if they ask about fundamentals, focus on P/E/market cap; if comparing assets, structure for comparison.
2. **Set up the Analysis section** — facts should serve as context that the Analysis section interprets, creating a natural narrative flow.
3. The `## Analysis` prompt was also refined: "directly address the user's question — don't just describe the data, explain what it means."

**Actions Taken**:
- Updated `SYNTHESIZE_ANALYSIS_PROMPT` in `src/financial_qa_agent/agent.py`:
  - `## Fact`: Replaced generic "include all data" with selective "present ONLY relevant data points" with examples per question type
  - `## Analysis`: Added "directly address the user's question" and "connect facts to a clear takeaway"
- Updated `specs/agent.md` — synthesize node: Fact section now described as "question-relevant factual data only"
- All 84 tests passing

---

## v0.0.17 — 2026-03-04 — Date-Range Queries Return All OHLCV Data Points

**Instruction**:
> I see that when asking for jan market activity for alibaba, the market data node output is very wrong which missing the first half, check why and fix

**Root Cause**:
`_fetch_ticker_data` in `market_data.py` always did `hist.tail(10)` — keeping only the last 10 data points regardless of query type. For period-based queries (e.g. `5d`), this was fine. But for date-range queries like January 1–February 1 (21 trading days), it chopped off the first 11 days, showing only Jan 16–30 instead of Jan 2–30.

**Design Decision**:
Date-range queries and period queries have different intent:
- **Date range** (`time_start`/`time_end`): user explicitly asked for that range — return **all** data points
- **Period** (`time_period`): user asked for a relative window — cap at **last 10** to keep LLM context manageable

**Actions Taken**:
- Updated `src/financial_qa_agent/tools/market_data.py`:
  - `_fetch_ticker_data`: removed `hist.tail(10)` entirely — all data points now pass through to the LLM
  - The synthesize prompt (updated in v0.0.16) already instructs the LLM to present only question-relevant facts
- Replaced two tests with one in `tests/test_tools.py`:
  - `test_market_data_returns_all_data_points` — verifies no server-side truncation (21 trading days all present)
- Updated `specs/agent.md` — fetch_market_data: no server-side truncation, LLM selects relevant facts
- All 85 tests passing (7 API + 32 agent + 28 models + 18 tools)

---

## v0.0.18 — 2026-03-04 — Commodity Ticker Support

**Instruction**:
> when ask "黄金过去一年的走势如何", it fails to extract tickers, you should instruct the parser to extract tickers not only for equities, but also commodities (should be related tickers available in yfinance)

**Actions Taken**:
- Updated `PARSE_PROMPT` in `src/financial_qa_agent/agent.py`:
  - Added commodity → yfinance futures mapping with both English and Chinese names (gold/黄金 → GC=F, silver/白银 → SI=F, crude oil/原油 → CL=F, Brent oil/布伦特原油 → BZ=F, natural gas/天然气 → NG=F, copper/铜 → HG=F, platinum/铂金 → PL=F, palladium/钯金 → PA=F, corn/玉米 → ZC=F, wheat/小麦 → ZW=F, soybean/大豆 → ZS=F, coffee/咖啡 → KC=F, sugar/糖 → SB=F, cotton/棉花 → CT=F)
  - Added `commodity` to `asset_type` enum
- Updated `specs/agent.md` — parse resolution and supported asset types now include commodities
- All 85 tests passing

---

## v0.0.19 — 2026-03-04 — Revert Period Summary, Instruct LLM to Compute from OHLCV

**Instruction**:
> when asking "阿里巴巴过去一年涨了多少", why the model is using the low price but not the price 1 y ago to do the computation

Initial fix added a computed period summary line to `_format_ticker_data`. User rejected this approach:

> I think you should not use a rule-based summary, or such fact will be endless to append, revert what you just did. instead I think you should instruct to analyzer you are just giving the raw ohlcv and some random field, the analyzer may need to extract necessary information from ohlcv itself not to only use provided field to answer question

**Design Decision**:
Instead of pre-computing metrics in Python (rule-based summaries that would grow endlessly for every possible question type), the synthesize prompt now explicitly tells the LLM:
1. The data includes raw OHLCV rows + some snapshot fields (current price, market cap, P/E, 52-week range)
2. Snapshot fields are point-in-time values, NOT historical — e.g. `52w_low` is NOT "the price one year ago"
3. To answer questions about price changes, trends, or performance, the LLM must compute from the OHLCV rows (compare first/last close, calculate % change, identify range highs/lows)

This keeps the data pipeline simple (raw data in, no pre-computation) and relies on the LLM's ability to read and compute from tabular data.

**Actions Taken**:
- Reverted `_format_ticker_data` in `src/financial_qa_agent/tools/market_data.py`:
  - Removed computed period summary line (start/end close, change, pct)
  - Back to simple OHLCV row listing
- Updated `SYNTHESIZE_ANALYSIS_PROMPT` `## Fact` section in `src/financial_qa_agent/agent.py`:
  - Explains that data includes raw OHLCV rows + snapshot fields
  - Warns that snapshot fields (52w range etc.) are NOT historical — don't misuse them
  - Instructs LLM to compute metrics from OHLCV rows (compare first/last close, calculate % change, etc.)
  - Still instructs to present only question-relevant computed facts, not dump all rows
- All 85 tests passing

---

## v0.0.20 — 2026-03-04 — Fix Missing References in Analysis with News

**Instruction**:
> when I ask "黄金最近两月涨的原因是什么", it did have needs_news=True but not have reference section? why? is it related to its asset type? find the cause and fix

**Root Cause**:
The `SYNTHESIZE_ANALYSIS_PROMPT` used `## User Question` and `## Available Data` as section headers for the input data. These `##` headers collided with the answer format headers (`## Fact`, `## Analysis`, `## References`), confusing the LLM about which `##` sections were its answer template vs. input data sections. Additionally, the `## References` instruction ("If news data is available...") appeared before the LLM saw the actual news data in `## Available Data`, making it harder to determine whether news data existed.

**Design Decision**:
Changed the input data sections from `## User Question` / `## Available Data` to `User Question:` / `Available Data:` (plain text with a `---` separator). This eliminates the heading-level collision — the LLM now clearly distinguishes between the `##` answer sections it should produce and the plain-text input data sections. Applied the same fix to `SYNTHESIZE_KNOWLEDGE_PROMPT` for consistency.

**Actions Taken**:
- Updated `SYNTHESIZE_ANALYSIS_PROMPT` in `src/financial_qa_agent/agent.py`:
  - Changed `## User Question` → `User Question:` and `## Available Data` → `Available Data:` with `---` separator
  - Made `## References` instruction more explicit: "Look at the 'Recent News' section in the available data below"
- Updated `SYNTHESIZE_KNOWLEDGE_PROMPT` — same heading-level fix for consistency
- All 85 tests passing

---

## v0.0.21 — 2026-03-04 — LLM-Determined Question Type + Frontend Label Rename

**Instructions**:
> 1. you should not use only whether there are tickers as a rule to distinguish analysis and knowledge, you should allow parse node to determine whether user query is analysis or knowledge check, maybe modify prompt to achieve this?
> 2. rename some key word, especially front end: market data → structured market data, news search → unstructured market data, local knowledge → knowledge hub, web knowledge → online knowledge

**Design Decisions**:
1. **LLM-determined `question_type`** — Previously, `question_type` was derived purely from ticker presence (`tickers → analysis`, no tickers → `knowledge`). This broke for questions like "黄金最近两月涨的原因是什么" where tickers exist (GC=F) but the user wants an explanation, not data analysis. Now the parse LLM explicitly classifies intent as "analysis" or "knowledge" with clear guidelines and examples in the prompt. A Pydantic validator coerces invalid values to "knowledge" (safe default).
2. **Three routing paths** — The routing function now uses `question_type` + ticker presence:
   - **Analysis** (tickers): fetch_market_data [+ fetch_news] — pure data analysis
   - **Knowledge with tickers**: fetch_market_data + fetch_local_knowledge [+ fetch_news] — market data as supporting context for the knowledge answer
   - **Knowledge without tickers**: fetch_local_knowledge only — pure knowledge Q/A
3. **Synthesize reads from parse_result** — The synthesize node now reads `parse_result.question_type` instead of inferring from `market_data` presence. This means a knowledge question with tickers (like "why did gold rise?") gets the knowledge prompt (`## Answer` + `## References`) even though market data is present in context.
4. **Frontend label rename** — Tab labels updated to more descriptive names:
   - "Market Data" → "Structured Market Data"
   - "News Search" → "Unstructured Market Data"
   - "Local Knowledge" → "Knowledge Hub"
   - "Web Knowledge" → "Online Knowledge"
   Internal tool names and tab IDs unchanged.

**Actions Taken**:
- Updated `src/financial_qa_agent/models.py`:
  - Added `question_type: str = "knowledge"` field to `ParseResultModel`
  - Added `validate_question_type` field validator (coerces invalid values to "knowledge")
- Updated `src/financial_qa_agent/agent.py`:
  - Added `question_type: str` to `ParseResult` TypedDict
  - Added `question_type` instruction to `PARSE_PROMPT` with intent definitions and examples
  - Added `question_type` to JSON output format in prompt
  - Rewrote `route_by_parse_result()` with 3 paths (analysis, knowledge+tickers, knowledge)
  - Updated `synthesize_node()` to read `parse_result.question_type` instead of deriving from `market_data`
  - Updated docstrings for `build_graph()`, `AgentState`, `route_by_parse_result()`
- Updated `frontend/index.html` — tab labels and placeholder text renamed
- Updated `frontend/style.css` — tab font reduced to 0.68rem, allow text wrap for longer labels
- Updated `tests/test_agent.py`:
  - Added `question_type` to all parse mock responses in pipeline tests
  - Rewrote 7 routing tests → 8 (added knowledge-with-tickers, knowledge-with-tickers-no-news)
  - Updated 3 synthesize tests to use `parse_result.question_type`
  - Added `test_synthesize_knowledge_with_market_data` — verifies knowledge prompt used despite market data
- Updated `tests/test_models.py`:
  - Added `test_question_type_analysis`, `test_question_type_knowledge`, `test_invalid_question_type_coerced_to_knowledge`
  - Updated `test_defaults` to check `question_type == "knowledge"`
- All 90 tests passing (7 API + 34 agent + 31 models + 18 tools)

---

## v0.0.22 — 2026-03-04 — Merge Knowledge Tabs + Shorten Tab Labels

**Instruction**:
> ok is too long. I want to refactor the tab again: structured market data -> market data, news -> news, knowledge hub and online knowledge -> merge these two tabs into one, but should separate the local and online, the local first and scroll down to see online

**Design Decision**:
Reduced the trace panel from 5 tabs to 4 by merging the two knowledge tabs into a single "Knowledge" tab. Within the merged tab, local and online results appear as labeled sub-sections (Local Knowledge first, Online Knowledge below with a separator). The `TOOL_TAB_MAP` maps both `local_knowledge` and `web_knowledge` SSE tool names to the same `tab-knowledge` pane; a `getToolContainer()` helper creates named sub-sections on first use.

**Actions Taken**:
- Updated `frontend/index.html` — 4 tabs: Agent Loop, Market Data, News, Knowledge (merged pane)
- Updated `frontend/app.js`:
  - `TOOL_TAB_MAP` maps both knowledge tools to `"tab-knowledge"`
  - Added `TOOL_SECTION_LABEL` for sub-section headers
  - Added `getToolContainer()` — creates/reuses named sub-sections within the Knowledge tab
  - `resetTrace()` clears 4 panes instead of 5
- Updated `frontend/style.css`:
  - Restored tab font to `0.75rem` / padding to `0.25rem` (4 tabs have more room)
  - Added `.knowledge-subsection`, `.subsection-header` styles (separator + blue uppercase label)
- All tests passing

---

## v0.0.23 — 2026-03-04 — Structured Fact Section: Key Metrics + Summary Bullet

**Instruction**:
> in the fact part of analysis, modify the prompt to have a clear structure to firstly have key fact metrics, and have a summary bullet, which answer user question in brief way, especially indicating whether the price movement is upward / downward or oscillating

**Design Decision**:
The `## Fact` section was previously a free-form paragraph that let the LLM decide how to present computed facts. This lacked consistent structure and often buried the headline answer. The new prompt enforces a two-part structure:
1. **Key Metrics** — A concise bullet list of computed question-relevant figures (start/end prices, % change, period high/low, etc.)
2. **Summary** — One sentence that directly answers the user's question in plain language. For price-movement questions, the LLM must explicitly state the trend direction: upward, downward, or sideways/oscillating, with the key number (e.g. "rose 12.3%").

This gives the reader a scannable metrics block and a glanceable answer before the deeper Analysis section.

**Actions Taken**:
- Updated `SYNTHESIZE_ANALYSIS_PROMPT` `## Fact` section in `src/financial_qa_agent/agent.py`:
  - Restructured into **Key Metrics** (bullet list of computed figures) + **Summary** (one-sentence answer with trend direction)
  - Summary must state upward/downward/oscillating for price questions, with key number
  - Retained existing instructions about OHLCV computation and snapshot field warnings
- All 90 tests passing

---
