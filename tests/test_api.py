"""Tests for the Financial QA Agent API."""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_agent():
    """Mock the agent to avoid real LLM/tool calls in API tests."""
    with patch("src.financial_qa_agent.main.agent") as mock:
        # Default ask_with_trace that emits a few events then sentinel
        async def _fake_trace(question, queue):
            await queue.put({"event_type": "trace", "stage": "parse", "status": "started", "detail": "Parsing..."})
            await queue.put({"event_type": "trace", "stage": "parse", "status": "completed", "detail": "Done"})
            await queue.put({"event_type": "answer", "answer": "Test trace answer."})
            await queue.put(None)  # sentinel
            return "Test trace answer."
        mock.ask_with_trace = AsyncMock(side_effect=_fake_trace)
        yield mock


@pytest.fixture
def client(mock_agent):
    """Async HTTP client with mocked agent."""
    from src.financial_qa_agent.main import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health():
    """Health endpoint works independently of the agent."""
    from src.financial_qa_agent.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# SSE streaming endpoint tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_content_type(client):
    """SSE endpoint returns text/event-stream content type."""
    resp = await client.post("/api/ask/stream", json={"question": "Test?"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


@pytest.mark.asyncio
async def test_stream_emits_events_and_done(client):
    """SSE stream contains trace events, an answer event, and ends with done."""
    resp = await client.post("/api/ask/stream", json={"question": "Test?"})
    body = resp.text

    # Should contain trace events
    assert "event: trace" in body
    # Should contain an answer event
    assert "event: answer" in body
    # Should always end with event: done
    assert "event: done" in body


@pytest.mark.asyncio
async def test_stream_error_event(client, mock_agent):
    """SSE stream emits error event when ask_with_trace raises."""
    async def _error_trace(question, queue):
        await queue.put({"event_type": "error", "message": "Something broke"})
        await queue.put(None)  # sentinel
        raise RuntimeError("Something broke")

    mock_agent.ask_with_trace = AsyncMock(side_effect=_error_trace)

    resp = await client.post("/api/ask/stream", json={"question": "Fail?"})
    body = resp.text

    assert "event: error" in body
    assert "Something broke" in body
    assert "event: done" in body
