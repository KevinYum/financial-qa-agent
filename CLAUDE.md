# Financial QA Agent - Project Rules

## Project Overview
A financial QA agent with a Python backend API and simple web frontend for demo purposes.

## Tech Stack
- **Backend**: Python, FastAPI, managed with `uv`
- **Agent**: LangGraph (StateGraph), langchain-openai (ChatOpenAI — works with OpenAI and OpenRouter)
- **Tools**: yfinance (market data), FMP (fundamental data), Brave Search API (news), ChromaDB (knowledge base)
- **Config**: pydantic-settings (reads `.env` automatically)
- **Frontend**: Vanilla HTML/CSS/JS (simple web SDK, no framework)
- **Package Manager**: `uv` (not pip, not poetry)

## Development Rules

### 1. Always Test After Changes
- After every code change, run tests before considering the task complete.
- Backend: `uv run pytest`
- Frontend: Open in browser / verify static files serve correctly
- API: Verify endpoints respond with `curl` or test client

### 2. Keep Rules, Specs, and README Updated
- When making changes, update related spec files in `specs/`, this `CLAUDE.md`, and `README.md`.
- Specs must reflect the current state of the system, not a past or future state.
- If a new feature is added, create or update the corresponding spec.
- `README.md` contains architecture visuals that must stay in sync:
  1. **System Architecture** (Mermaid) — components, data flow, interaction pattern
  2. **Agent Loop** (text diagram) — question processing pipeline
  3. **Project Structure** (text tree) — folder layout and file responsibilities
- When adding files, endpoints, middleware, or changing the agent pipeline, update the relevant section(s).

### 3. Instruction Tracking
- All user instructions are logged in `docs/instructions.md` with timestamps and version numbers.
- Every prompt that drives project direction must be recorded.

### 4. Code Standards
- Python: Follow PEP 8, use type hints, keep functions focused.
- Frontend: Keep it simple — vanilla JS, no build step.
- API responses: Always return JSON with consistent structure.

### 5. Project Structure
```
financial-qa-agent/
├── README.md              # Project docs with architecture diagrams
├── CLAUDE.md              # This file - project rules
├── pyproject.toml         # Python project config (uv)
├── .env.example           # Template for environment variables
├── src/
│   └── financial_qa_agent/
│       ├── __init__.py
│       ├── config.py      # Settings (pydantic-settings, reads .env)
│       ├── main.py        # FastAPI app entry point
│       ├── agent.py       # LangGraph agent pipeline
│       ├── models.py      # Pydantic models for tools, parse result, trace events
│       └── tools/
│           ├── __init__.py
│           ├── market_data.py        # yfinance OHLCV + fundamentals
│           ├── fundamental_data.py   # FMP API: financial statements + earnings transcripts
│           ├── news_search.py        # Brave Search API
│           └── knowledge_base.py     # ChromaDB local search + Brave web search
├── tests/
│   ├── conftest.py        # Shared test fixtures
│   ├── test_api.py        # API endpoint tests
│   ├── test_agent.py      # LangGraph pipeline tests
│   ├── test_models.py     # Pydantic model unit tests
│   └── test_tools.py      # Tool unit tests
├── frontend/
│   ├── index.html         # Two-panel layout + marked.js/DOMPurify CDN
│   ├── style.css          # Grid layout, markdown styles, trace entries
│   └── app.js             # SSE streaming, markdown rendering, trace panel
├── specs/
│   ├── api.md             # API specification
│   ├── architecture.md    # Architecture overview
│   └── agent.md           # Agent loop specification
├── data/
│   └── chroma/            # ChromaDB persistence (gitignored)
└── docs/
    └── instructions.md    # Instruction log with timestamps
```

### 6. Running the Project
- Backend: `uv run uvicorn src.financial_qa_agent.main:app --reload --port 8000`
- Frontend: Open `frontend/index.html` in browser (or serve via backend static files)
- Tests: `uv run pytest`

### 7. API Design
- Batch endpoint: `POST /api/ask` — receives a financial question, returns an answer (JSON envelope).
- Streaming endpoint: `POST /api/ask/stream` — SSE stream with progressive trace events + final answer.
- All batch responses follow: `{"status": "ok"|"error", "data": ..., "message": "..."}`
- SSE events: `trace`, `tool_input`, `tool_output`, `answer`, `error`, `done`
