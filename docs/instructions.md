# Instruction Log

All user instructions that drive project direction are recorded here with timestamps and version numbers.

---

## v0.0.1 — 2026-03-04T10:00:00Z — Project Initialization

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

## v0.0.2 — 2026-03-04T10:30:00Z — README with Architecture Diagrams

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

## v0.0.3 — 2026-03-04T11:00:00Z — Project Structure Fix + Agent Loop Design

**Instruction**:
> 1. Project structure should be text-based like `folderA/... --- what is the content or file usage`, not in diagram.
> 2. Let's start design the agent loop. Do you think it's a good idea to orchestrate agent loop using LangGraph SDK, cuz it will be a multi-step fetch and then use all content fetched in different steps to generate the final answer through an LLM call.

**Actions Taken**:
- Replaced Mermaid project structure diagram in `README.md` with a text-based tree using `---` annotations
- Updated `CLAUDE.md` rule #2 to note diagram #3 is text-based (not Mermaid)
- Discussed LangGraph SDK as agent orchestrator — design decision pending

---

## v0.0.4 — 2026-03-04T12:00:00Z — LangGraph Agent Loop Implementation

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

## v0.0.5 — 2026-03-04T13:00:00Z — Documentation Cleanup

**Instruction**:
> 1. Instruction log is missing major design decisions made during the agent loop implementation (v0.4.0).
> 2. Replace the README agent loop Mermaid diagram with a text-based diagram (same style as the project structure section — using box-drawing characters, not Mermaid).

**Actions Taken**:
- Added **Design Decisions** section to v0.0.4 entry documenting: LangGraph choice, classify-route-fetch-synthesize pattern, 4-way classification, pluggable LLM, tool design rationale, shared state approach, interface preservation
- Replaced agent loop Mermaid diagram in `README.md` with text-based box-drawing diagram
- Logged this instruction

---

## v0.0.6 — 2026-03-04T14:00:00Z — Versioning Fix + ChromaDB Embedding Clarification

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

## v0.0.7 — 2026-03-04T16:00:00Z — Parse-Driven Routing (Remove Classification)

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
