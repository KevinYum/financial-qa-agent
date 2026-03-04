# Architecture Specification

**Version**: 0.5.0
**Last Updated**: 2026-03-04

## Overview
A financial QA agent with a LangGraph-orchestrated pipeline, two-type question routing (analysis vs. knowledge), three data-fetching tools, SSE-based trace streaming, and a vanilla web frontend with a two-panel layout (chat + trace).

## Components

### Backend (Python / FastAPI)
- **Entry point**: `src/financial_qa_agent/main.py`
- **Framework**: FastAPI with uvicorn
- **Role**: Serves the QA API, SSE streaming endpoint, and static frontend files
- **Config**: `src/financial_qa_agent/config.py` — pydantic-settings, reads `.env`
- **Logging**: DEBUG level by default — shows agent trace in console

### Agent (LangGraph)
- **File**: `src/financial_qa_agent/agent.py`
- **Orchestrator**: LangGraph `StateGraph` with 5 nodes
- **LLM**: `langchain-openai` `ChatOpenAI` — works with OpenAI and OpenRouter via `base_url`
- **State**: `AgentState(TypedDict)` — question, parse_result, question_type, market_data, news_data, knowledge_data, answer
- **Pipeline**: parse → two-type route (analysis: market_data [+ news] | knowledge: knowledge_base) → synthesize → return
- **Question types**: analysis (tickers present → data summary + analysis) | knowledge (no tickers → answer + references)
- **Tracing**: `contextvars.ContextVar`-based trace queue; nodes emit events via `_emit_trace()` / `_emit_trace_sync()`

### Data Models (Pydantic)
- **File**: `src/financial_qa_agent/models.py`
- **Purpose**: Centralized Pydantic models for all structured data flowing through the pipeline
- **Market data**: `HistoryRecord` (OHLCV), `TickerData` (fundamentals + history, with `52w_high`/`52w_low` alias handling)
- **Knowledge base**: `KnowledgeResult` (text + source + title)
- **News search**: `NewsResult` (title, description, url, age)
- **Parse validation**: `ParseResultModel` — validates LLM JSON output with defaults, companion to `ParseResult` TypedDict. Includes `field_validator` that coerces invalid `time_period` values to `None` (safety net for LLM hallucinations like `"2w"`).
- **Trace events**: `TraceEvent`, `ToolInputEvent`, `ToolOutputEvent`, `AnswerEvent`, `ErrorEvent` — typed event models for SSE protocol
- **Key design**: LangGraph requires `TypedDict` state; Pydantic models serve as validation companions (construct → `.model_dump()` → state)

### Tools (`src/financial_qa_agent/tools/`)
1. **market_data.py** — yfinance: OHLCV, fundamentals (tickers from parse node, no regex fallback). Constructs `TickerData` and `HistoryRecord` models, returns `.model_dump(by_alias=True)`.
2. **news_search.py** — Brave Search API: recent financial news. Constructs `NewsResult` models for structured formatting.
3. **knowledge_base.py** — ChromaDB vector search with Brave web fallback, auto-population, URL/title metadata retention. Constructs `KnowledgeResult` models for results.

All tools include DEBUG-level logging for backend observability.

### Frontend (Vanilla HTML/CSS/JS)
- **Location**: `frontend/`
- **Purpose**: Demo UI with two-panel layout — chat (left) + trace (right)
- **No build step** — plain HTML, CSS, and JS files
- **Communicates with backend via SSE stream (`POST /api/ask/stream`)**
- **Trace panel**: 4 tabs — Agent Loop, Market Data, News Search, Knowledge Base
- **Progressive updates**: SSE events update the trace panel in real-time during pipeline execution

## External Services
- **LLM API** (OpenAI or OpenRouter) — used for parse + synthesize nodes
- **Brave Search API** — used by news_search tool and knowledge_base fallback
- **Yahoo Finance** — used by market_data tool (no API key required)

## Local Storage
- **ChromaDB** — persistent vector DB at `data/chroma/` for knowledge base

## Data Flow
```
User (browser) → app.js → POST /api/ask/stream → FastAPI → LangGraph agent:
  parse (LLM) → route by question type:
    Analysis (tickers): fetch_market_data [+ fetch_news] → synthesize (data + analysis)
    Knowledge (no tickers): fetch_knowledge              → synthesize (answer + references)
  → SSE events → browser

  Trace events flow: agent nodes → asyncio.Queue → SSE generator → browser
  Chat panel: answer event → rendered as agent message
  Trace panel: trace/tool_input/tool_output events → rendered in respective tabs
```

## Deployment
- Local development only (localhost)
- Backend on port 8000
- Frontend served as static files by the backend

## Diagrams
Visual architecture diagrams are maintained in [`README.md`](../README.md):
1. System Architecture (Mermaid) — full component & data flow with external services
2. Agent Loop (text diagram) — parse → route → fetch → synthesize pipeline
3. Project Structure (text tree) — file layout & responsibilities
