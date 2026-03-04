# Financial QA Agent - Project Rules

## Project Overview
A financial QA agent with a Python backend API and simple web frontend for demo purposes.

## Tech Stack
- **Backend**: Python, FastAPI, managed with `uv`
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
- `README.md` contains 3 Mermaid architecture diagrams that must stay in sync:
  1. **System Architecture** — components, data flow, interaction pattern
  2. **Agent Loop** — question processing pipeline
  3. **Project Structure** — folder layout and file responsibilities
- When adding files, endpoints, middleware, or changing the agent pipeline, update the relevant diagram(s).

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
├── src/
│   └── financial_qa_agent/
│       ├── __init__.py
│       ├── main.py        # FastAPI app entry point
│       └── agent.py       # QA agent logic
├── tests/
│   └── test_api.py        # API tests
├── frontend/
│   ├── index.html         # Demo web page
│   ├── style.css
│   └── app.js
├── specs/
│   ├── api.md             # API specification
│   └── architecture.md    # Architecture overview
└── docs/
    └── instructions.md    # Instruction log with timestamps
```

### 6. Running the Project
- Backend: `uv run uvicorn src.financial_qa_agent.main:app --reload --port 8000`
- Frontend: Open `frontend/index.html` in browser (or serve via backend static files)
- Tests: `uv run pytest`

### 7. API Design
- Single product endpoint: `POST /api/ask` — receives a financial question, returns an answer.
- All responses follow: `{"status": "ok"|"error", "data": ..., "message": "..."}`
