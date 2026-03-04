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
