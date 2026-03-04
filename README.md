# Financial QA Agent

A financial question-answering agent with a Python/FastAPI backend and a vanilla web frontend for demo purposes.

## Quick Start

```bash
# Install dependencies
uv sync

# Copy and fill in environment variables
cp .env.example .env
# Edit .env with your API keys (LLM_API_KEY, BRAVE_API_KEY)

# Start the server (backend + frontend)
uv run uvicorn src.financial_qa_agent.main:app --reload --port 8000

# Open in browser
open http://localhost:8000

# Run tests
uv run pytest -v
```

---

## Architecture

### 1. System Architecture — Components, Data Flow & Interaction

```mermaid
flowchart TB
    subgraph Browser["Browser (localhost:8000)"]
        UI["index.html\nChat + Trace UI"]
        JS["app.js\nSSE Client"]
    end

    subgraph Backend["FastAPI Backend (Python)"]
        direction TB
        UV["uvicorn\nASGI Server\n:8000"]

        subgraph Middleware["Middleware"]
            CORS["CORS\nMiddleware"]
            STATIC["StaticFiles\nMiddleware"]
        end

        subgraph Routes["API Routes"]
            HEALTH["GET /health"]
            ASK["POST /api/ask"]
            STREAM["POST /api/ask/stream\n(SSE)"]
        end

        subgraph Models["Pydantic Models"]
            REQ["AskRequest\n{question}"]
            RES["APIResponse\n{status, data, message}"]
        end

        subgraph AgentCore["LangGraph Agent (agent.py)"]
            TRACE["Trace Queue\n(contextvars)"]
            PARSE["parse\n(LLM)"]
            FETCH_MD["fetch_market_data\n(yfinance)"]
            FETCH_NEWS["fetch_news\n(Brave API)"]
            FETCH_LOCAL["fetch_local_knowledge\n(ChromaDB)"]
            EVAL_SUFF["evaluate_sufficiency\n(LLM)"]
            FETCH_WEB["fetch_web_knowledge\n(Brave → ChromaDB)"]
            SYNTH["synthesize\n(LLM)"]
        end
    end

    subgraph External["External Services"]
        LLM_API["LLM API\n(OpenAI / OpenRouter)"]
        BRAVE["Brave Search API"]
        YAHOO["Yahoo Finance"]
    end

    subgraph Local["Local Storage"]
        CHROMA["ChromaDB\n(data/chroma/)"]
    end

    %% User interaction
    USER((User)) -- "types question" --> UI
    UI --> JS
    JS -- "POST /api/ask/stream" --> UV

    %% Request flow
    UV --> CORS --> STREAM
    STATIC -- "static files" --> Browser

    %% Agent flow
    STREAM -- "validate" --> REQ
    REQ --> PARSE
    PARSE --> FETCH_MD & FETCH_NEWS & FETCH_LOCAL
    FETCH_MD & FETCH_NEWS --> SYNTH
    FETCH_LOCAL --> EVAL_SUFF
    EVAL_SUFF -. "sufficient" .-> SYNTH
    EVAL_SUFF -. "insufficient" .-> FETCH_WEB
    FETCH_WEB --> SYNTH
    PARSE & FETCH_MD & FETCH_NEWS & FETCH_LOCAL & EVAL_SUFF & FETCH_WEB & SYNTH -. "trace events" .-> TRACE
    TRACE -. "SSE stream" .-> JS
    JS -- "render chat + trace" --> UI

    %% External connections
    PARSE & SYNTH -. "LLM calls" .-> LLM_API
    FETCH_MD -. "OHLCV + info" .-> YAHOO
    FETCH_NEWS -. "web search" .-> BRAVE
    FETCH_LOCAL -. "vector search" .-> CHROMA
    FETCH_WEB -. "web search" .-> BRAVE
    FETCH_WEB -. "store results" .-> CHROMA

    style Browser fill:#e8f4fd,stroke:#0066ff
    style Backend fill:#f0f0f0,stroke:#333
    style Middleware fill:#fff3cd,stroke:#856404
    style Routes fill:#d4edda,stroke:#155724
    style Models fill:#f8d7da,stroke:#721c24
    style AgentCore fill:#cce5ff,stroke:#004085
    style External fill:#f5e6ff,stroke:#6f42c1
    style Local fill:#e2e3e5,stroke:#495057
```

**Interaction Pattern**: The frontend sends `POST /api/ask/stream` via `fetch()`. FastAPI validates through Pydantic, then launches the LangGraph agent with a trace queue. The agent parses the question into structured entities (tickers, time period, news flag, knowledge queries), then routes based on question type: **analysis** (tickers found → market data + optional news → sectioned answer: Fact + Analysis + References) or **knowledge hub** (no tickers → local ChromaDB first → LLM evaluates sufficiency → if insufficient, web search → sectioned answer: Answer + References). The LLM produces markdown with `##` section headers; the frontend renders it via `marked.js` (sanitized with DOMPurify) so headers, links, lists, bold text, tables, and code all display correctly. Each pipeline stage emits trace events to an `asyncio.Queue`, which the SSE generator streams to the browser in real-time. The frontend renders trace events in a 5-tab trace panel (Agent Loop, Market Data, News Search, Local Knowledge, Web Knowledge) and the final markdown answer in the chat panel. A batch endpoint (`POST /api/ask`) is also available for backward compatibility.

---

### 2. Agent Loop — Question Processing Pipeline

```
User Question
     │
     ▼
┌─────────────┐
│    parse     │  ← LLM call: extract tickers, time period, news flag, knowledge queries
└──────┬──────┘
       │  (route by question type — determined by tickers)
       │
       ├── tickers found? ─── TYPE A: ANALYSIS ──────────────────────────────────────────┐
       │   └── fetch_market_data   (yfinance OHLCV + fundamentals)                       │
       │   └── [+ fetch_news]      (Brave Search API, if needs_news)                     │
       │                                                                                  │
       ├── no tickers? ──── TYPE B: KNOWLEDGE HUB ──────────────────────────────────┐    │
       │   └── fetch_local_knowledge     (ChromaDB vector search)                    │    │
       │          │                                                                  │    │
       │          ▼                                                                  │    │
       │   ┌───────────────────────┐                                                 │    │
       │   │ evaluate_sufficiency   │  ← LLM: are local results enough?              │    │
       │   └──────────┬────────────┘                                                 │    │
       │              │                                                              │    │
       │      ┌───────┴────────┐                                                     │    │
       │      │                │                                                     │    │
       │   sufficient    insufficient                                                │    │
       │      │                │                                                     │    │
       │      │       fetch_web_knowledge  (Brave → store in ChromaDB)               │    │
       │      │                │                                                     │    │
       ▼      ▼                ▼                                                     ▼    ▼
┌─────────────┐                                                              ┌─────────────┐
│ synthesize   │  ← Knowledge prompt (## Answer + ## References)             │ synthesize   │
│ (knowledge)  │                                                             │ (analysis)   │
└──────┬──────┘                                                              └──────┬──────┘
       │       ← Analysis prompt (## Fact + ## Analysis + ## References)            │
       ▼                                                                            ▼
┌─────────────┐
│  response    │  ← Frontend parses ## sections, renders clickable links
└─────────────┘
```

**Two question types**: The presence of tickers in the parse result determines the question type. **Analysis** questions (tickers found) fetch market data and optionally news, producing a sectioned answer: **Fact** (structured data) + **Analysis** (interpretation) + **References** (news links, if available). **Knowledge** questions (no tickers) use a **knowledge hub** approach: local ChromaDB is queried first, then an LLM evaluates whether the local results sufficiently answer the question — if not, a web search is triggered and results are stored in ChromaDB for future use. The final answer is a sectioned response: **Answer** (explanation) + **References** (source links). The LLM outputs markdown; the frontend renders it natively via `marked.js` with DOMPurify sanitization.

---

### 3. Project Structure

```
financial-qa-agent/
├── README.md                       --- Project documentation with architecture diagrams
├── CLAUDE.md                       --- Development rules and conventions for Claude
├── pyproject.toml                  --- uv project config, dependencies, pytest settings
├── uv.lock                         --- Locked dependency versions
├── .env.example                    --- Template for required environment variables
├── .gitignore                      --- Git ignore rules
│
├── src/financial_qa_agent/         --- Backend Python package
│   ├── __init__.py                 --- Package marker
│   ├── config.py                   --- Settings via pydantic-settings (LLM, Brave, ChromaDB)
│   ├── main.py                     --- FastAPI app: routes, Pydantic models, static mount
│   ├── agent.py                    --- LangGraph agent: parse → fetch → synthesize pipeline
│   ├── models.py                   --- Pydantic models: TickerData, NewsResult, trace events, etc.
│   └── tools/                      --- Data fetching tool modules
│       ├── __init__.py             --- Tool exports
│       ├── market_data.py          --- yfinance: OHLCV, fundamentals (tickers from parse node)
│       ├── news_search.py          --- Brave Search API: recent financial news
│       └── knowledge_base.py       --- Local ChromaDB search + web search with auto-population
│
├── frontend/                       --- Vanilla web UI (no build step)
│   ├── index.html                  --- Two-panel layout + marked.js/DOMPurify CDN
│   ├── style.css                   --- Grid layout, markdown styles, trace entries
│   └── app.js                      --- SSE streaming, markdown rendering, trace panel
│
├── tests/                          --- Test suite (all externals mocked)
│   ├── __init__.py
│   ├── conftest.py                 --- Shared fixtures (mock LLM responses)
│   ├── test_api.py                 --- API endpoint + SSE stream tests (7 tests)
│   ├── test_agent.py               --- LangGraph pipeline + routing + trace tests (32 tests)
│   ├── test_models.py              --- Pydantic model unit tests (28 tests)
│   └── test_tools.py               --- Tool unit tests (18 tests)
│
├── specs/                          --- Living specifications
│   ├── api.md                      --- Endpoint contracts and response format
│   ├── architecture.md             --- Component overview and data flow
│   └── agent.md                    --- Agent loop design, state schema, tool interfaces
│
├── data/                           --- Runtime data (gitignored)
│   └── chroma/                     --- ChromaDB persistent vector storage
│
└── docs/                           --- Project history
    └── instructions.md             --- Timestamped log of every user instruction
```

---

## API Reference

| Method | Endpoint            | Description                          |
|--------|---------------------|--------------------------------------|
| POST   | `/api/ask`          | Submit a financial question (batch)  |
| POST   | `/api/ask/stream`   | Submit with SSE trace streaming      |
| GET    | `/health`           | Health check                         |

**Request** (`POST /api/ask` or `POST /api/ask/stream`):
```json
{ "question": "What is compound interest?" }
```

**Batch Response** (`/api/ask`):
```json
{
  "status": "ok",
  "data": {
    "question": "What is compound interest?",
    "answer": "..."
  },
  "message": "Question answered successfully"
}
```

**SSE Stream** (`/api/ask/stream`): Returns `text/event-stream` with events: `trace`, `tool_input`, `tool_output`, `answer`, `error`, `done`. See [`specs/api.md`](specs/api.md) for full SSE event reference.

---

## Development

| Command | Purpose |
|---------|---------|
| `uv sync` | Install / update dependencies |
| `uv run uvicorn src.financial_qa_agent.main:app --reload --port 8000` | Start dev server |
| `uv run pytest -v` | Run test suite |

### Project Conventions
- **Rules**: See [`CLAUDE.md`](CLAUDE.md) for all development rules
- **Specs**: See [`specs/`](specs/) for API and architecture specifications
- **Instruction log**: See [`docs/instructions.md`](docs/instructions.md) for full history

---

## Tech Stack

- **Backend**: Python 3.13+, FastAPI, uvicorn
- **Agent Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) (StateGraph)
- **LLM Provider**: [langchain-openai](https://python.langchain.com/docs/integrations/chat/openai/) (OpenAI / OpenRouter)
- **Market Data**: [yfinance](https://github.com/ranaroussi/yfinance) (OHLCV, fundamentals)
- **News Search**: [Brave Search API](https://brave.com/search/api/)
- **Knowledge Base**: [ChromaDB](https://www.trychroma.com/) (local vector DB with web fallback)
- **Frontend**: Vanilla HTML / CSS / JS (no build step)
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **Testing**: pytest, pytest-asyncio, httpx
