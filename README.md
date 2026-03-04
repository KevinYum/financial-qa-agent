# Financial QA Agent

A financial question-answering agent with a Python/FastAPI backend and a vanilla web frontend for demo purposes.

## Quick Start

```bash
# Install dependencies
uv sync

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
        CSS["style.css"]
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
            REQ["AskRequest\n{question: str}"]
            RES["APIResponse\n{status, data, message}"]
        end

        subgraph Core["Agent Core"]
            AGENT["FinancialQAAgent\nagent.py"]
        end
    end

    %% User interaction
    USER((User)) -- "types question" --> UI
    UI --> JS
    JS -- "POST /api/ask\n{question: ...}" --> UV

    %% Request flow
    UV --> CORS --> STATIC
    STATIC -- "/*.html,css,js" --> Browser
    CORS --> ASK
    UV --> HEALTH

    %% Processing flow
    ASK -- "validate" --> REQ
    REQ --> AGENT
    AGENT -- "answer" --> RES
    RES -- "JSON response" --> JS
    JS -- "render answer" --> UI

    style Browser fill:#e8f4fd,stroke:#0066ff
    style Backend fill:#f0f0f0,stroke:#333
    style Middleware fill:#fff3cd,stroke:#856404
    style Routes fill:#d4edda,stroke:#155724
    style Models fill:#f8d7da,stroke:#721c24
    style Core fill:#cce5ff,stroke:#004085
```

**Interaction Pattern**: The frontend is a single-page chat UI that sends `POST /api/ask` requests via `fetch()`. The backend validates the input through Pydantic, delegates to the agent, and returns a JSON envelope `{status, data, message}`. The same uvicorn process serves both the API and the static frontend files.

---

### 2. Agent Loop — Question Processing Pipeline

```mermaid
flowchart TD
    START((User submits\nquestion)) --> VALIDATE

    subgraph FastAPI["FastAPI Route Handler"]
        VALIDATE{"Pydantic\nValidation"}
        VALIDATE -- "invalid\n(missing field)" --> ERR422["422 Validation Error"]
    end

    VALIDATE -- "valid\nAskRequest" --> AGENT_ENTRY

    subgraph AgentLoop["Agent Processing Loop (agent.py)"]
        AGENT_ENTRY["agent.ask(question)"]
        EMPTY_CHECK{"Empty or\nwhitespace?"}
        AGENT_ENTRY --> EMPTY_CHECK

        EMPTY_CHECK -- "yes" --> RAISE["raise ValueError"]

        EMPTY_CHECK -- "no" --> PROCESS

        subgraph PROCESS["Processing Pipeline (TBD)"]
            direction TB
            STEP1["1. Parse & Classify\nthe question"]
            STEP2["2. Retrieve Context\n(knowledge / tools)"]
            STEP3["3. Generate Answer\n(LLM / rule engine)"]
            STEP4["4. Format Response"]
            STEP1 --> STEP2 --> STEP3 --> STEP4
        end

        PROCESS --> ANSWER["Return answer string"]
    end

    RAISE --> CATCH_VAL
    ANSWER --> BUILD_OK

    subgraph ResponseBuilder["Response Builder"]
        BUILD_OK["APIResponse\nstatus=ok\ndata={question, answer}"]
        CATCH_VAL["APIResponse\nstatus=error\nmessage=ValueError text"]
        CATCH_EXC["APIResponse\nstatus=error\nmessage=Internal server error"]
    end

    BUILD_OK --> JSON_OUT
    CATCH_VAL --> JSON_OUT
    CATCH_EXC --> JSON_OUT

    JSON_OUT(("JSON to\nbrowser"))

    style FastAPI fill:#d4edda,stroke:#155724
    style AgentLoop fill:#cce5ff,stroke:#004085
    style PROCESS fill:#fff3cd,stroke:#856404,stroke-dasharray: 5 5
    style ResponseBuilder fill:#f8d7da,stroke:#721c24
```

> **Note**: The inner "Processing Pipeline" steps (parse, retrieve, generate, format) are placeholders. The agent currently returns a stub response. The actual behavior — LLM integration, tool use, retrieval strategy — will be defined in a future iteration.

---

### 3. Project Structure — Files & Responsibilities

```mermaid
flowchart LR
    subgraph Root["financial-qa-agent/"]
        README["README.md\nProject documentation\n& architecture diagrams"]
        CLAUDE["CLAUDE.md\nDevelopment rules\nfor Claude"]
        PYPROJECT["pyproject.toml\nuv config, deps,\npytest settings"]
        GITIGNORE[".gitignore"]
    end

    subgraph SrcPkg["src/financial_qa_agent/"]
        INIT["__init__.py\nPackage marker"]
        MAIN["main.py\nFastAPI app\nRoutes, models,\nstatic mount"]
        AGENTPY["agent.py\nFinancialQAAgent\nQuestion processing"]
    end

    subgraph FrontDir["frontend/"]
        HTML["index.html\nChat UI shell"]
        CSSF["style.css\nLayout & theme"]
        APPJS["app.js\nfetch() client\nDOM rendering"]
    end

    subgraph TestDir["tests/"]
        TINIT["__init__.py"]
        TAPI["test_api.py\n4 tests: health,\nask, empty, missing"]
    end

    subgraph SpecDir["specs/"]
        APISPEC["api.md\nEndpoint contracts\n& response format"]
        ARCHSPEC["architecture.md\nComponent overview\n& data flow"]
    end

    subgraph DocDir["docs/"]
        INSTRLOG["instructions.md\nTimestamped log of\nevery user instruction"]
    end

    %% Relationships
    MAIN -- "imports" --> AGENTPY
    MAIN -- "serves" --> FrontDir
    APPJS -- "calls" --> MAIN
    TAPI -- "tests" --> MAIN

    style Root fill:#f5f5f5,stroke:#333
    style SrcPkg fill:#cce5ff,stroke:#004085
    style FrontDir fill:#e8f4fd,stroke:#0066ff
    style TestDir fill:#d4edda,stroke:#155724
    style SpecDir fill:#fff3cd,stroke:#856404
    style DocDir fill:#f8d7da,stroke:#721c24
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
- **Frontend**: Vanilla HTML / CSS / JS (no build step)
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **Testing**: pytest, pytest-asyncio, httpx
