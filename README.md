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
        UI["index.html\nChat UI"]
        JS["app.js\nFetch Client"]
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
        end

        subgraph Models["Pydantic Models"]
            REQ["AskRequest\n{question}"]
            RES["APIResponse\n{status, data, message}"]
        end

        subgraph AgentCore["LangGraph Agent (agent.py)"]
            PARSE["parse\n(LLM)"]
            FETCH_MD["fetch_market_data\n(yfinance)"]
            FETCH_NEWS["fetch_news\n(Brave API)"]
            FETCH_KB["fetch_knowledge\n(ChromaDB)"]
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
    JS -- "POST /api/ask" --> UV

    %% Request flow
    UV --> CORS --> ASK
    STATIC -- "static files" --> Browser

    %% Agent flow
    ASK -- "validate" --> REQ
    REQ --> PARSE
    PARSE --> FETCH_MD & FETCH_NEWS & FETCH_KB
    FETCH_MD & FETCH_NEWS & FETCH_KB --> SYNTH
    SYNTH --> RES
    RES -- "JSON" --> JS
    JS -- "render" --> UI

    %% External connections
    PARSE & SYNTH -. "LLM calls" .-> LLM_API
    FETCH_MD -. "OHLCV + info" .-> YAHOO
    FETCH_NEWS -. "web search" .-> BRAVE
    FETCH_KB -. "vector search" .-> CHROMA
    FETCH_KB -. "fallback search" .-> BRAVE

    style Browser fill:#e8f4fd,stroke:#0066ff
    style Backend fill:#f0f0f0,stroke:#333
    style Middleware fill:#fff3cd,stroke:#856404
    style Routes fill:#d4edda,stroke:#155724
    style Models fill:#f8d7da,stroke:#721c24
    style AgentCore fill:#cce5ff,stroke:#004085
    style External fill:#f5e6ff,stroke:#6f42c1
    style Local fill:#e2e3e5,stroke:#495057
```

**Interaction Pattern**: The frontend sends `POST /api/ask` via `fetch()`. FastAPI validates through Pydantic, then delegates to the LangGraph agent. The agent parses the question into structured entities (tickers, time period, news flag, knowledge queries), routes to one or more fetch tools based on what was extracted, and synthesizes a final answer via an LLM call. The response returns as a JSON envelope `{status, data, message}`.

---

### 2. Agent Loop — Question Processing Pipeline

```
User Question
     │
     ▼
┌─────────────┐
│    parse     │  ← LLM call: extract tickers, time period, news flag, knowledge queries
└──────┬──────┘
       │  (route by parse result — no classification label)
       │
       ├── tickers found?        ──► fetch_market_data   (yfinance OHLCV + fundamentals)
       ├── needs_news = true?    ──► fetch_news          (Brave Search API)
       ├── knowledge_queries?    ──► fetch_knowledge     (ChromaDB lookup → if sparse, Brave search → save to ChromaDB)
       └── nothing extracted?    ──► fetch_knowledge     (fallback)
       │
       │  (multiple tools fire in parallel when multiple fields are populated)
       ▼
┌─────────────┐
│ synthesize   │  ← LLM call with parse result + ALL fetched context + original question → final answer
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  response    │  ← Structure into API envelope {status, data, message}
└─────────────┘
```

**Routing is data-driven**: tool selection depends on which parse result fields are populated, not a classification label. Multiple tools can fire in parallel.

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
│   └── tools/                      --- Data fetching tool modules
│       ├── __init__.py             --- Tool exports
│       ├── market_data.py          --- yfinance: OHLCV, fundamentals, ticker extraction
│       ├── news_search.py          --- Brave Search API: recent financial news
│       └── knowledge_base.py       --- ChromaDB vector search + Brave web fallback
│
├── frontend/                       --- Vanilla web UI (no build step)
│   ├── index.html                  --- Chat interface shell
│   ├── style.css                   --- Layout and theme
│   └── app.js                      --- fetch() client, DOM rendering
│
├── tests/                          --- Test suite (all externals mocked)
│   ├── __init__.py
│   ├── conftest.py                 --- Shared fixtures (mock LLM responses)
│   ├── test_api.py                 --- API endpoint tests (4 tests)
│   ├── test_agent.py               --- LangGraph pipeline integration tests (15 tests)
│   └── test_tools.py               --- Tool unit tests (21 tests)
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

| Method | Endpoint     | Description                  |
|--------|-------------|------------------------------|
| POST   | `/api/ask`  | Submit a financial question  |
| GET    | `/health`   | Health check                 |

**Request** (`POST /api/ask`):
```json
{ "question": "What is compound interest?" }
```

**Response**:
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

All responses follow the envelope: `{ status, data, message }`

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
