"""Tests for the Financial QA Agent API."""

import pytest
from unittest.mock import AsyncMock, patch
from httpx import ASGITransport, AsyncClient


@pytest.fixture
def mock_agent():
    """Mock the agent to avoid real LLM/tool calls in API tests."""
    with patch("src.financial_qa_agent.main.agent") as mock:
        mock.ask = AsyncMock(return_value="This is a test answer.")
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


@pytest.mark.asyncio
async def test_ask_success(client):
    resp = await client.post("/api/ask", json={"question": "What is inflation?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["data"]["question"] == "What is inflation?"
    assert body["data"]["answer"] == "This is a test answer."
    assert body["message"] == "Question answered successfully"


@pytest.mark.asyncio
async def test_ask_empty_question(client, mock_agent):
    mock_agent.ask = AsyncMock(side_effect=ValueError("Question cannot be empty"))
    resp = await client.post("/api/ask", json={"question": "   "})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "error"
    assert body["data"] is None


@pytest.mark.asyncio
async def test_ask_missing_question():
    """Missing question field triggers Pydantic validation error."""
    from src.financial_qa_agent.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        resp = await c.post("/api/ask", json={})
    assert resp.status_code == 422  # Pydantic validation error
