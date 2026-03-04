"""FastAPI application entry point for the Financial QA Agent."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .agent import FinancialQAAgent

app = FastAPI(title="Financial QA Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = FinancialQAAgent()

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


class AskRequest(BaseModel):
    question: str


class AskResponseData(BaseModel):
    question: str
    answer: str


class APIResponse(BaseModel):
    status: str
    data: AskResponseData | None = None
    message: str


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask", response_model=APIResponse)
async def ask(request: AskRequest) -> APIResponse:
    try:
        answer = await agent.ask(request.question)
        return APIResponse(
            status="ok",
            data=AskResponseData(question=request.question, answer=answer),
            message="Question answered successfully",
        )
    except ValueError as e:
        return APIResponse(status="error", data=None, message=str(e))
    except Exception:
        return APIResponse(
            status="error", data=None, message="Internal server error"
        )


# Serve frontend static files
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
