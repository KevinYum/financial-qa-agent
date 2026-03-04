# Architecture Specification

**Version**: 0.1.0
**Last Updated**: 2026-03-04

## Overview
A simple financial QA agent consisting of a Python backend API and a vanilla web frontend.

## Components

### Backend (Python / FastAPI)
- **Entry point**: `src/financial_qa_agent/main.py`
- **Framework**: FastAPI with uvicorn
- **Role**: Serves the QA API and optionally static frontend files
- **Agent logic**: `src/financial_qa_agent/agent.py` — processes financial questions

### Frontend (Vanilla HTML/CSS/JS)
- **Location**: `frontend/`
- **Purpose**: Demo UI to interact with the QA agent
- **No build step** — plain HTML, CSS, and JS files
- **Communicates with backend via fetch to `POST /api/ask`**

## Data Flow
```
User (browser) → frontend/app.js → POST /api/ask → FastAPI → agent.py → response → browser
```

## Deployment
- Local development only (localhost)
- Backend on port 8000
- Frontend served as static files by the backend

## Diagrams
Visual architecture diagrams (Mermaid) are maintained in [`README.md`](../README.md):
1. System Architecture — full component & data flow
2. Agent Loop — question processing pipeline
3. Project Structure — file layout & responsibilities
