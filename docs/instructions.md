# Instruction Log

All user instructions that drive project direction are recorded here with timestamps and version numbers.

---

## v0.1.0 — 2026-03-04T10:00:00Z — Project Initialization

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

## v0.2.0 — 2026-03-04T10:30:00Z — README with Architecture Diagrams

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
