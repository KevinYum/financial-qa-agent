# Architecture Specification

**Version**: 0.2.0
**Last Updated**: 2026-03-04

## Overview
A financial QA agent with a LangGraph-orchestrated pipeline, three data-fetching tools, and a vanilla web frontend.

## Components

### Backend (Python / FastAPI)
- **Entry point**: `src/financial_qa_agent/main.py`
- **Framework**: FastAPI with uvicorn
- **Role**: Serves the QA API and static frontend files
- **Config**: `src/financial_qa_agent/config.py` — pydantic-settings, reads `.env`

### Agent (LangGraph)
- **File**: `src/financial_qa_agent/agent.py`
- **Orchestrator**: LangGraph `StateGraph` with 5 nodes
- **LLM**: `langchain-openai` `ChatOpenAI` — works with OpenAI and OpenRouter via `base_url`
- **State**: `AgentState(TypedDict)` — question, classification, market_data, news_data, knowledge_data, answer
- **Pipeline**: classify → conditional route → fetch tool(s) → synthesize → return

### Tools (`src/financial_qa_agent/tools/`)
1. **market_data.py** — yfinance: OHLCV, fundamentals, ticker extraction
2. **news_search.py** — Brave Search API: recent financial news
3. **knowledge_base.py** — ChromaDB vector search with Brave web fallback and auto-population

### Frontend (Vanilla HTML/CSS/JS)
- **Location**: `frontend/`
- **Purpose**: Demo UI to interact with the QA agent
- **No build step** — plain HTML, CSS, and JS files
- **Communicates with backend via fetch to `POST /api/ask`**

## External Services
- **LLM API** (OpenAI or OpenRouter) — used for classify + synthesize nodes
- **Brave Search API** — used by news_search tool and knowledge_base fallback
- **Yahoo Finance** — used by market_data tool (no API key required)

## Local Storage
- **ChromaDB** — persistent vector DB at `data/chroma/` for knowledge base

## Data Flow
```
User (browser) → app.js → POST /api/ask → FastAPI → LangGraph agent:
  classify (LLM) → route → fetch tools → synthesize (LLM) → response → browser
```

## Deployment
- Local development only (localhost)
- Backend on port 8000
- Frontend served as static files by the backend

## Diagrams
Visual architecture diagrams are maintained in [`README.md`](../README.md):
1. System Architecture (Mermaid) — full component & data flow with external services
2. Agent Loop (Mermaid) — classify → route → fetch → synthesize pipeline
3. Project Structure (text tree) — file layout & responsibilities
