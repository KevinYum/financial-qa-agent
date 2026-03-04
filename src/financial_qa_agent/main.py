"""FastAPI application entry point for the Financial QA Agent."""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from .agent import FinancialQAAgent

# ---------------------------------------------------------------------------
# Logging — default to DEBUG so agent trace is visible in console
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Financial QA Agent", version="0.2.0")

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


# ---------------------------------------------------------------------------
# SSE streaming endpoint — progressive trace events
# ---------------------------------------------------------------------------


@app.post("/api/ask/stream")
async def ask_stream(request: AskRequest):
    """Stream agent pipeline trace events as Server-Sent Events.

    Uses ``agent.ask_with_trace`` which pushes trace events to an
    ``asyncio.Queue``.  Each event is yielded in SSE wire format::

        event: <type>
        data: <json>\n\n

    The stream always ends with an ``event: done`` message.
    """
    trace_queue: asyncio.Queue = asyncio.Queue()

    async def event_generator():
        task = asyncio.create_task(
            agent.ask_with_trace(request.question, trace_queue)
        )
        try:
            while True:
                event = await trace_queue.get()
                if event is None:  # sentinel — stream done
                    break
                event_type = event.pop("event_type", "trace")
                yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
        except Exception as exc:
            logger.exception("SSE stream error")
            error_payload = json.dumps({"message": str(exc)})
            yield f"event: error\ndata: {error_payload}\n\n"
        finally:
            # Ensure task finishes / is cancelled
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
            yield f"event: done\ndata: {{}}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Static frontend files (must be last — catch-all mount)
# ---------------------------------------------------------------------------

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
